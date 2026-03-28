mod analyzer;
mod atlas;
mod geo;
mod io;
mod model;

use anyhow::{bail, Result};
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use rstar::RTree;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;

use analyzer::AnycastAnalyzer;
use atlas::{fetch_atlas_measurement, parse_atlas_id};
use io::{decompress_gz, load_airports, load_input_data, EMBEDDED_AIRPORTS, EMBEDDED_CITIES};
use model::{Airport, Disc, OutputRecord};

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "Input CSV file (mutually exclusive with --atlas)")]
    input: Option<PathBuf>,

    #[arg(short, long, help = "Output CSV file (defaults to atlas_<ID>.csv when using --atlas)")]
    output: Option<PathBuf>,

    #[arg(
        long,
        help = "RIPE Atlas measurement ID or URL (mutually exclusive with --input)"
    )]
    atlas: Option<String>,

    #[arg(
        short,
        long,
        default_value = "cities",
        help = "Dataset to use: 'cities' (embedded), 'airports' (embedded), or path to custom CSV"
    )]
    dataset: String,

    #[arg(
        short,
        long,
        default_value_t = 0,
        help = "Absolute minimum population threshold (filter cities at load time)"
    )]
    min_pop: u32,

    #[arg(
        short = 'p',
        long,
        default_value_t = 0.0,
        help = "Relative population threshold (0.0-1.0): keep cities with pop >= max_pop * ratio within each geolocation"
    )]
    pop_ratio: f32,

    #[arg(
        short,
        long,
        default_value_t = 1.0,
        help = "Alpha (population vs distance score tuning)"
    )]
    alpha: f32,

    #[arg(
        short,
        long,
        default_value_t = 0,
        help = "Discard disks with RTT > threshold"
    )]
    threshold: u32,

    #[arg(
        long,
        default_value_t = false,
        help = "Only output anycast geolocations (skip unicast)"
    )]
    anycast: bool,
}

fn main() -> Result<()> {
    // Parse arguments
    let args = Args::parse();

    if args.input.is_some() && args.atlas.is_some() {
        bail!("--input and --atlas are mutually exclusive. Use one or the other.");
    }
    if args.input.is_none() && args.atlas.is_none() {
        bail!("Either --input or --atlas must be provided.");
    }

    // Get RIPE Atlas ID
    let atlas_id = match &args.atlas {
        Some(atlas_input) => Some(parse_atlas_id(atlas_input)?),
        None => None,
    };

    // Get output path and filename
    let output_path = match args.output {
        Some(p) => p,
        None => match atlas_id {
            Some(id) => PathBuf::from(format!("atlas_{}.csv", id)),
            None => bail!("--output is required when using --input."),
        },
    };

    // Load in the airports/cities (optionally filtered by minimum population)
    let airports = match args.dataset.as_str() {
        "airports" => {
            println!("Using embedded airports dataset.");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_AIRPORTS)?), 0)?
        }
        "cities" => {
            let filters: Vec<String> = [
                (args.min_pop > 0).then(|| format!("min pop: {}", args.min_pop)),
                (args.pop_ratio > 0.0).then(|| format!("relative: {}×max", args.pop_ratio)),
            ]
            .into_iter()
            .flatten()
            .collect();

            if filters.is_empty() {
                println!("Using embedded cities dataset.");
            } else {
                println!("Using embedded cities dataset ({}).", filters.join(", "));
            }
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES)?), args.min_pop)?
        }
        custom_path => {
            println!("Loading custom dataset from: {}", custom_path);
            load_airports(File::open(custom_path)?, args.min_pop)?
        }
    };
    println!("Loaded {} locations.", airports.len());

    // Build spatial r-tree for the locations
    println!("Building spatial index...");
    let airport_tree: RTree<Airport> = RTree::bulk_load(airports);
    println!(
        "Spatial index ready ({} locations indexed).",
        airport_tree.size()
    );

    // Load input data (file or RIPE Atlas measurement)
    let in_df = if let Some(ref input_path) = args.input {
        println!("Loading input data from: {:?}", input_path);
        load_input_data(input_path, args.threshold)?
    } else {
        fetch_atlas_measurement(atlas_id.unwrap(), args.threshold)?
    };

    // Create a group for each target address
    let groups_df = in_df.group_by(["addr"])?.groups()?;
    // Get the indices for each group
    let group_indices = groups_df.column("groups")?.list()?;

    let num_targets = group_indices.len();
    println!(
        "Starting parallel processing for {} targets...",
        num_targets
    );

    // Create progress bar
    let pb = ProgressBar::new(num_targets as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    // Perform geolocation
    let results: Vec<OutputRecord> = group_indices
        .into_iter() // Iterate over each series of indices
        .par_bridge() // Bridge to Rayon's multi-thread processing
        .progress_with(pb) // Progress bar
        .filter_map(|opt_indices_series| {
            opt_indices_series.map(|indices_series| { // Perform for each individual indices series
                // Get the dataframe data of this indices group
                let indices_ca = indices_series.u32().unwrap();
                let group_df = in_df.take(indices_ca).unwrap();

                // Get each column as a group of values (typed arrays)
                let target = group_df.column("addr").unwrap().str().unwrap();
                let hostname = group_df.column("hostname").unwrap().str().unwrap();
                let lat_rad = group_df.column("lat_rad").unwrap().f32().unwrap();
                let lon_rad = group_df.column("lon_rad").unwrap().f32().unwrap();
                let radius = group_df.column("radius").unwrap().f32().unwrap();

                // Get a reference to the first target value (all the same)
                let target_arc: Arc<str> = Arc::from(target.get(0).unwrap_or(""));

                // Build a Disc per measurement
                let discs: Vec<Disc> = (0..group_df.height())
                    .map(|i| Disc {
                        target: Arc::clone(&target_arc),
                        hostname: hostname.get(i).unwrap_or("").to_string(),
                        lat: lat_rad.get(i).unwrap_or(0.0),
                        lon: lon_rad.get(i).unwrap_or(0.0),
                        radius: radius.get(i).unwrap_or(0.0),
                    })
                    .collect();

                // Run algorithm on the discs for this target, and return output
                let analyzer = AnycastAnalyzer::new(
                    discs,
                    &airport_tree,
                    args.alpha,
                    args.pop_ratio,
                    args.anycast,
                );
                analyzer.analyze()
            })
        })
        .flatten()
        .collect();

    println!(
        "Analysis complete. Found {} geolocated sites (unicast + anycast).",
        results.len()
    );

    // Write results to path
    if !results.is_empty() {
        println!("Saving results to {:?}...", output_path);
        let num_results = results.len();
        let mut output_df = DataFrame::new(num_results, vec![
            Series::new(
                "addr".into(),
                results.iter().map(|r| &*r.target).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "vp".into(),
                results.iter().map(|r| r.vp.as_str()).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "vp_lat".into(),
                results.iter().map(|r| r.vp_lat).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "vp_lon".into(),
                results.iter().map(|r| r.vp_lon).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "radius".into(),
                results.iter().map(|r| r.radius).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "pop_iata".into(),
                results
                    .iter()
                    .map(|r| r.pop_iata.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "pop_lat".into(),
                results.iter().map(|r| r.pop_lat).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "pop_lon".into(),
                results.iter().map(|r| r.pop_lon).collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "pop_city".into(),
                results
                    .iter()
                    .map(|r| r.pop_city.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
            Series::new(
                "pop_cc".into(),
                results
                    .iter()
                    .map(|r| r.pop_cc.as_str())
                    .collect::<Vec<_>>(),
            )
            .into(),
        ])?;

        let mut file = File::create(output_path)?;
        CsvWriter::new(&mut file)
            .with_separator(b'\t')
            .finish(&mut output_df)?;

        println!("Results successfully saved.");
    } else {
        println!("No geolocated sites found, no output file written.");
    }

    Ok(())
}
