use anyhow::{Result, bail};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use crate::geo::{FIBER_RI, SPEED_OF_LIGHT};
use crate::model::Airport;
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
    let mut lazy = df.lazy().filter(
        col("rtt")
            .is_not_null()
            .and(col("rtt").gt(lit(0.0f32)))
    );

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

/// Read the input CSV.
///
/// Input columns must contain `addr,hostname,rtt` and optionally `lat,lon`.
/// If the latter is missing, a VPs file must be supplied (mapping hostnames to `lat,lon` values).
pub fn load_input_data(path: &PathBuf, threshold: u32, vps: Option<&VpTable>) -> Result<DataFrame> {
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

    let input_file = File::open(path)?;
    let df = CsvReader::new(input_file)
        .with_options(read_options)
        .finish()?;

    // Optionally get coordinates from a vps file
    let df = match vps {
        Some(vps) => attach_vp_coordinates(df, vps)?,
        None => df,
    };

    finalize_measurements(df, threshold)
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
