use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;
use std::sync::Arc;

use crate::geo::{FIBER_RI, SPEED_OF_LIGHT};
use crate::model::{Airport, OutputRecord};
use crate::vps::{Vp, VpTable};

pub static EMBEDDED_AIRPORTS: &[u8] = include_bytes!("../datasets/airports.csv.gz");
pub static EMBEDDED_CITIES: &[u8] = include_bytes!("../datasets/cities500.csv.gz");

pub fn decompress_gz(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

/// The progress bar used for every long parallel phase.
pub fn progress_bar(len: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

pub fn load_airports<R: polars::io::mmap::MmapBytesReader>(
    reader: R,
    min_pop: u32,
) -> Result<Vec<Airport>> {
    let airports_schema = Arc::new(Schema::from_iter([
        Field::new(PlSmallStr::from("iata"), DataType::String),
        Field::new(PlSmallStr::from("size"), DataType::String),
        Field::new(PlSmallStr::from("name"), DataType::String),
        Field::new(PlSmallStr::from("country_code"), DataType::String),
        Field::new(PlSmallStr::from("city"), DataType::String),
        Field::new(PlSmallStr::from("lat"), DataType::Float32),
        Field::new(PlSmallStr::from("lon"), DataType::Float32),
        Field::new(PlSmallStr::from("pop"), DataType::UInt32),
        Field::new(PlSmallStr::from("heuristic"), DataType::Int32),
    ]));

    let airports_read_options = CsvReadOptions {
        has_header: true,
        schema: Some(airports_schema),
        parse_options: Arc::new(CsvParseOptions::default().with_separator(b'\t')),
        ..Default::default()
    };

    let mut airports_df = CsvReader::new(reader)
        .with_options(airports_read_options)
        .finish()?
        .lazy();

    if min_pop > 0 {
        airports_df = airports_df.filter(col("pop").gt_eq(lit(min_pop)));
    }

    let airports_df = airports_df
        .with_columns([
            col("lat").radians().alias("lat_rad"),
            col("lon").radians().alias("lon_rad"),
        ])
        .collect()?;

    let iata = airports_df.column("iata")?.str()?;
    let lat = airports_df.column("lat")?.f32()?;
    let lon = airports_df.column("lon")?.f32()?;
    let pop = airports_df.column("pop")?.u32()?;
    let city = airports_df.column("city")?.str()?;
    let country_code = airports_df.column("country_code")?.str()?;
    let lat_rad = airports_df.column("lat_rad")?.f32()?;
    let lon_rad = airports_df.column("lon_rad")?.f32()?;

    let airports: Vec<Airport> = (0..airports_df.height())
        .map(|i| Airport {
            iata: iata.get(i).unwrap_or("").to_string(),
            lat: lat.get(i).unwrap_or(0.0),
            lon: lon.get(i).unwrap_or(0.0),
            pop: pop.get(i).unwrap_or(0),
            city: city.get(i).unwrap_or("").to_string(),
            country_code: country_code.get(i).unwrap_or("").to_string(),
            lat_rad: lat_rad.get(i).unwrap_or(0.0),
            lon_rad: lon_rad.get(i).unwrap_or(0.0),
        })
        .collect();

    Ok(airports)
}

/// Turn raw measurements into the discs the algorithm consumes.
///
/// Non-positive and missing RTTs are skipped.
/// Each vantage point contributes one disc per target (the lowest RTT).
///
/// Input: `addr`, `hostname`, `lat`, `lon` and `rtt` (in ms)
/// Output: Adds the columns `lat_rad`, `lon_rad` and `radius` (km).
pub fn finalize_measurements(df: DataFrame, threshold: u32) -> Result<DataFrame> {
    let before = df.height();
    // Drop negative and NaN RTT measurements
    let mut lazy = df
        .lazy()
        .filter(col("rtt").is_not_null().and(col("rtt").gt(lit(0.0f32))));

    if threshold > 0 {
        lazy = lazy.filter(col("rtt").lt_eq(lit(threshold as f32)));
    }

    // Get the minimum RTT per VP (order-preserving grouping for consistency between runs)
    let deduped = lazy
        .group_by_stable([col("addr"), col("hostname")])
        .agg([col("lat").first(), col("lon").first(), col("rtt").min()])
        .with_columns([
            col("lat").radians().alias("lat_rad"),
            col("lon").radians().alias("lon_rad"),
            (col("rtt") * lit(0.001) * lit(SPEED_OF_LIGHT) / lit(FIBER_RI) / lit(2.0))
                .alias("radius"),
        ])
        .collect()?;

    if deduped.height() == 0 {
        bail!("No usable measurements remain after filtering.");
    }

    println!(
        "Using {} measurements ({} dropped by filtering and deduplication).",
        deduped.height(),
        before.saturating_sub(deduped.height())
    );

    Ok(deduped)
}

/// Columns that may carry the vantage point's identity.
/// `hostname` - expected
/// `rx` - for compatability with MAnycastR
const VP_COLUMNS: [&str; 2] = ["hostname", "rx"];

/// Read the measurements at `path`, as CSV or as Parquet.
///
/// The format is taken from the extension: `.parquet` is read as Parquet
/// ([`load_parquet_data`]), anything else as CSV ([`load_csv_data`]), which
/// decompresses the file first when it ends in `.gz`.
pub fn load_input_data(path: &Path, threshold: u32, vps: Option<&VpTable>) -> Result<DataFrame> {
    if is_parquet(path) {
        load_parquet_data(path, threshold, vps)
    } else {
        load_csv_data(path, threshold, vps)
    }
}

/// Read the input CSV.
///
/// Input columns must contain `addr,hostname,rtt` and optionally `lat,lon`.
/// If the latter is missing, a VPs file must be supplied (mapping hostnames to `lat,lon` values).
fn load_csv_data(path: &Path, threshold: u32, vps: Option<&VpTable>) -> Result<DataFrame> {
    let mut fields = vec![
        Field::new(PlSmallStr::from("addr"), DataType::String),
        Field::new(PlSmallStr::from("hostname"), DataType::String),
    ];
    // If there is no VPs file, extract lat/lon values from the input CSV
    if vps.is_none() {
        fields.push(Field::new(PlSmallStr::from("lat"), DataType::Float32));
        fields.push(Field::new(PlSmallStr::from("lon"), DataType::Float32));
    }
    fields.push(Field::new(PlSmallStr::from("rtt"), DataType::Float32));

    let read_options = CsvReadOptions {
        has_header: true,
        schema: Some(Arc::new(Schema::from_iter(fields))),
        ..Default::default()
    };

    let df = if is_gzipped(path) {
        CsvReader::new(Cursor::new(gunzip(path)?))
            .with_options(read_options)
            .finish()?
    } else {
        let input_file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        CsvReader::new(input_file)
            .with_options(read_options)
            .finish()?
    };

    // Optionally get coordinates from a vps file
    let df = match vps {
        Some(vps) => attach_vp_coordinates(df, vps)?,
        None => df,
    };

    finalize_measurements(df, threshold)
}

/// Read measurements from a Parquet file.
fn load_parquet_data(path: &Path, threshold: u32, vps: Option<&VpTable>) -> Result<DataFrame> {
    let file = File::open(path)
        .with_context(|| format!("failed to open Parquet file {}", path.display()))?;
    let schema = ParquetReader::new(file).schema()?;
    let has = |name: &str| schema.iter_names().any(|field| field.as_str() == name);

    let Some(vp_column) = VP_COLUMNS.into_iter().find(|name| has(name)) else {
        bail!(
            "{} has no vantage point column: expected one of {}.",
            path.display(),
            VP_COLUMNS.join(", ")
        );
    };
    for required in ["addr", "rtt"] {
        if !has(required) {
            bail!("{} has no '{required}' column.", path.display());
        }
    }

    // Coordinates in the file are only used when no VPs file overrides them.
    let use_file_coordinates = vps.is_none() && has("lat") && has("lon");
    if vps.is_none() && !use_file_coordinates {
        bail!(
            "{} has no 'lat'/'lon' columns; supply --vps to resolve {vp_column} to coordinates.",
            path.display()
        );
    }

    // Columns we want
    let mut wanted = vec!["addr".to_string(), vp_column.to_string(), "rtt".to_string()];
    if use_file_coordinates {
        wanted.push("lat".to_string());
        wanted.push("lon".to_string());
    }

    let file = File::open(path)
        .with_context(|| format!("failed to open Parquet file {}", path.display()))?;
    let mut df = ParquetReader::new(file)
        .with_columns(Some(wanted))
        .finish()?;

    // Addresses may be stored as text or as packed bytes; the algorithm wants text.
    let addr = df.column("addr")?.as_materialized_series();
    if addr.dtype() != &DataType::String {
        let decoded = decode_packed_addresses(addr)
            .with_context(|| format!("could not read the 'addr' column of {}", path.display()))?;
        df.with_column(decoded.into_column())?;
    }

    let mut df = df
        .lazy()
        .with_columns([
            col(vp_column).cast(DataType::String).alias("hostname"),
            col("rtt").cast(DataType::Float32),
        ])
        .collect()?;

    if use_file_coordinates {
        df = df
            .lazy()
            .with_columns([
                col("lat").cast(DataType::Float32),
                col("lon").cast(DataType::Float32),
            ])
            .collect()?;
    }

    let df = match vps {
        Some(vps) => attach_vp_coordinates(df, vps)?,
        None => df,
    };

    finalize_measurements(df, threshold)
}

/// Turn a column of packed binary addresses into printable text.
fn decode_packed_addresses(series: &Series) -> Result<Series> {
    let binary = series.cast(&DataType::Binary).map_err(|_| {
        anyhow::anyhow!(
            "'addr' is {} which is neither text nor packed address bytes",
            series.dtype()
        )
    })?;

    let addresses: Vec<Option<String>> = binary
        .binary()?
        .iter()
        .map(|value| value.and_then(format_packed_address))
        .collect();

    Ok(Series::new("addr".into(), addresses))
}

/// Render packed address bytes as text.
///
/// MAnycastR stores every address as 16 IPv6-mapped bytes, so an IPv4 target arrives
/// as `::ffff:1.1.1.1`; it is unwrapped back to `1.1.1.1`. Plain 4-byte IPv4 is
/// accepted too. Anything else yields `None`, dropping the row.
fn format_packed_address(bytes: &[u8]) -> Option<String> {
    match bytes.len() {
        4 => {
            let octets: [u8; 4] = bytes.try_into().ok()?;
            Some(Ipv4Addr::from(octets).to_string())
        }
        16 => {
            let octets: [u8; 16] = bytes.try_into().ok()?;
            let address = Ipv6Addr::from(octets);
            Some(match address.to_ipv4_mapped() {
                Some(v4) => v4.to_string(),
                None => address.to_string(),
            })
        }
        _ => None,
    }
}

/// Pack a textual address into 16 bytes, IPv4 as IPv6-mapped (`::ffff:1.1.1.1`).
fn pack_address(addr: &str) -> Option<[u8; 16]> {
    match addr.parse::<IpAddr>().ok()? {
        IpAddr::V4(v4) => Some(v4.to_ipv6_mapped().octets()),
        IpAddr::V6(v6) => Some(v6.octets()),
    }
}

/// Half the Earth's circumference (km).
const MAX_DISTANCE_KM: f32 = 20_038.0;

/// Report a distance in whole kilometres, capped at [`MAX_DISTANCE_KM`].
fn distance_km(value: f32) -> u16 {
    value.clamp(0.0, MAX_DISTANCE_KM).round() as u16
}

/// Rows per Parquet row group.
const OUTPUT_ROW_GROUP_SIZE: usize = 256 * 1024;

/// Write the results to `path`: Parquet when it ends in `.parquet`, tab-separated CSV otherwise.
///
/// Rows are sorted by `addr` (then PoP and VP), for identical output between runs.
/// Parquet stores `addr` as 16 packed bytes and leaves missing values null.
/// CSV writes `addr` as string, "NoCity"/"N/A" for a missing location.
pub fn write_results(results: Vec<OutputRecord>, path: &Path, accuracy: bool) -> Result<()> {
    let parquet = is_parquet(path);

    // Convert all addresses (string) to packed addresses
    let mut keyed: Vec<(Option<[u8; 16]>, OutputRecord)> = results
        .into_iter()
        .map(|r| (pack_address(&r.target), r))
        .collect();
    // Sort for stable output, better compression, and organized row groups
    keyed.sort_unstable_by(|(key_a, a), (key_b, b)| {
        key_a
            .cmp(key_b)
            .then_with(|| a.target.cmp(&b.target))
            .then_with(|| a.pop_iata.cmp(&b.pop_iata))
            .then_with(|| a.vp.cmp(&b.vp))
    });
    let rows = || keyed.iter().map(|(_, r)| r);

    let addr = if parquet {
        // Disallow non-IP addresses for .parquet output
        if let Some((_, r)) = keyed.iter().find(|(key, _)| key.is_none()) {
            bail!(
                "Cannot write '{}' (invalid IP address), write as .csv or ensure valid IP address formats.",
                r.target
            );
        }
        Series::new(
            "addr".into(),
            keyed
                .iter()
                .map(|(key, _)| key.as_ref().map(|bytes| bytes.as_slice()))
                .collect::<Vec<_>>(),
        )
    } else {
        Series::new(
            "addr".into(),
            rows().map(|r| &*r.target).collect::<Vec<_>>(),
        )
    };

    // Create columns as Series with column name
    let mut columns: Vec<Column> = vec![
        addr.into(),
        Series::new(
            "vp".into(),
            rows().map(|r| r.vp.as_str()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "vp_lat".into(),
            rows().map(|r| r.vp_lat).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "vp_lon".into(),
            rows().map(|r| r.vp_lon).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "radius".into(),
            rows().map(|r| distance_km(r.radius)).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "pop_iata".into(),
            rows().map(|r| r.pop_iata.as_deref()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "pop_lat".into(),
            rows().map(|r| r.pop_lat).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "pop_lon".into(),
            rows().map(|r| r.pop_lon).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "pop_city".into(),
            rows().map(|r| r.pop_city.as_deref()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "pop_cc".into(),
            rows().map(|r| r.pop_cc.as_deref()).collect::<Vec<_>>(),
        )
        .into(),
    ];

    // Append accuracy columns if --accuracy flag is set
    if accuracy {
        columns.push(
            Series::new(
                "candidate_diameter".into(),
                rows()
                    .map(|r| r.candidate_diameter.map(distance_km))
                    .collect::<Vec<_>>(),
            )
            .into(),
        );
        columns.push(
            Series::new(
                "num_constraints".into(),
                rows().map(|r| r.num_constraints).collect::<Vec<_>>(),
            )
            .into(),
        );
    }

    // Create dataframe from filled columns
    let mut df = DataFrame::new(keyed.len(), columns)?;
    // Create output file
    let mut file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;

    if parquet {
        ParquetWriter::new(&mut file)
            .with_row_group_size(Some(OUTPUT_ROW_GROUP_SIZE))
            .finish(&mut df)?;
    } else {
        // Replace null with NoCity and N/A
        let mut fills = vec![
            col("pop_iata").fill_null(lit("NoCity")),
            col("pop_city").fill_null(lit("N/A")),
            col("pop_cc").fill_null(lit("N/A")),
        ];
        if accuracy {
            fills.push(col("candidate_diameter").fill_null(lit(0u16)));
            fills.push(col("num_constraints").fill_null(lit(0u16)));
        }
        // Write csv
        let mut df = df.lazy().with_columns(fills).collect()?;
        CsvWriter::new(&mut file)
            .with_separator(b'\t')
            .finish(&mut df)?;
    }

    Ok(())
}

/// Whether the path names a .parquet file, by its extension.
fn is_parquet(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
}

/// Whether the path names a gzipped file, by its extension as `--warts` does.
fn is_gzipped(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
}

/// Decompress a gzipped file into memory.
fn gunzip(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut decompressed = Vec::new();
    GzDecoder::new(BufReader::with_capacity(1 << 20, file))
        .read_to_end(&mut decompressed)
        .with_context(|| format!("failed to decompress {}", path.display()))?;
    Ok(decompressed)
}

/// Add `lat`/`lon` columns by resolving each row's `hostname` against the VPs file.
///
/// Rows with a hostname missing from the VPs file are skipped.
fn attach_vp_coordinates(df: DataFrame, vps: &VpTable) -> Result<DataFrame> {
    let hostnames = df.column("hostname")?.str()?;

    // Resolution is per unique hostname
    let mut cache: HashMap<&str, Option<&Vp>> = HashMap::new();
    let mut canonical: Vec<Option<&str>> = Vec::with_capacity(df.height());
    let mut lats: Vec<Option<f32>> = Vec::with_capacity(df.height());
    let mut lons: Vec<Option<f32>> = Vec::with_capacity(df.height());
    let mut unknown: Vec<String> = Vec::new();

    for i in 0..df.height() {
        let vp = match hostnames.get(i) {
            Some(name) => *cache.entry(name).or_insert_with(|| {
                let found = vps.get(name);
                if found.is_none() {
                    unknown.push(name.to_string());
                }
                found
            }),
            None => None,
        };
        canonical.push(vp.map(|v| v.hostname.as_str()));
        lats.push(vp.map(|v| v.lat));
        lons.push(vp.map(|v| v.lon));
    }

    if !unknown.is_empty() {
        unknown.sort();
        unknown.dedup();
        let shown: Vec<&str> = unknown.iter().take(5).map(|s| s.as_str()).collect();
        let more = unknown.len().saturating_sub(shown.len());
        println!(
            "Dropped rows for {} vantage point(s) absent from the VPs file: {}{}.",
            unknown.len(),
            shown.join(", "),
            if more > 0 {
                format!(", and {more} more")
            } else {
                String::new()
            }
        );
    }

    let mut df = df;
    df.with_column(Series::new("hostname".into(), canonical).into())?;
    df.with_column(Series::new("lat".into(), lats).into())?;
    df.with_column(Series::new("lon".into(), lons).into())?;
    Ok(df
        .lazy()
        .filter(col("lat").is_not_null().and(col("lon").is_not_null()))
        .collect()?)
}
