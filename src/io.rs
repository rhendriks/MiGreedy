use anyhow::Result;
use flate2::read::GzDecoder;
use polars::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use crate::geo::{FIBER_RI, SPEED_OF_LIGHT};
use crate::model::Airport;

pub static EMBEDDED_AIRPORTS: &[u8] = include_bytes!("../datasets/airports.csv.gz");
pub static EMBEDDED_CITIES: &[u8] = include_bytes!("../datasets/cities500.csv.gz");

pub fn decompress_gz(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
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

pub fn load_input_data(path: &PathBuf, threshold: u32) -> Result<DataFrame> {
    let input_schema = Arc::new(Schema::from_iter([
        Field::new(PlSmallStr::from("addr"), DataType::String),
        Field::new(PlSmallStr::from("hostname"), DataType::String),
        Field::new(PlSmallStr::from("lat"), DataType::Float32),
        Field::new(PlSmallStr::from("lon"), DataType::Float32),
        Field::new(PlSmallStr::from("rtt"), DataType::Float32),
    ]));

    let read_options = CsvReadOptions {
        has_header: true,
        schema: Some(input_schema),
        ..Default::default()
    };

    let input_file = File::open(path)?;

    let mut in_df = CsvReader::new(input_file)
        .with_options(read_options)
        .finish()?
        .lazy();

    if threshold > 0 {
        in_df = in_df.filter(col("rtt").lt_eq(lit(threshold as f32)));
    }

    in_df = in_df.with_columns([
        col("lat").radians().alias("lat_rad"),
        col("lon").radians().alias("lon_rad"),
        (col("rtt") * lit(0.001) * lit(SPEED_OF_LIGHT) / lit(FIBER_RI) / lit(2.0)).alias("radius"),
    ]);

    let in_df = in_df.collect()?;

    println!(
        "Loaded {} latency measurements after applying RTT threshold filter.",
        in_df.height()
    );

    Ok(in_df)
}
