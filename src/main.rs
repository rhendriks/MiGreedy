//! # MiGreedy
//!
//! A fast, parallel improved version of the [iGreedy](https://github.com/fp7mplane/demo-infra/tree/master/igreedy)
//! algorithm for large-scale anycast-aware geolocation and is
//! used to produce daily anycast censuses, [publicly available](https://github.com/ut-dacs/anycast-census).
//!
//! Given latency (RTT) measurements from geographically dispersed vantage points,
//! MiGreedy detects whether a target is anycast and geolocates its sites:
//!
//! * Each measurement is turned into a disc bounding the target's possible location
//!   (Great-Circle-Distance, see [`geo`]).
//! * Discs that cannot cover the same location are collected into a maximum
//!   independent set (MIS); more than one MIS disc means the target is anycast.
//! * Each MIS cluster is geolocated to the best city (or airport) inside the
//!   intersection of its discs, scored on population and distance (see [`analyzer`]).
//!
//! Measurements must contain the following columns `addr, hostname, rtt`.
//! They also need `lat, lon` which can be additional columns
//! or extracted using a VPs file (`--vps`, see [`vps`]).
//!
//! * a CSV file (`--input`, see [`io`]);
//! * scamper warts files read natively (`--warts`, see [`warts`]).
//!
//! We also support RIPE Atlas measurements fetched over the API (`--atlas`, see [`atlas`]);
//! and scheduling new ones (`--measure`).
//!
//! Results are written as CSV (see [`io`]).
mod analyzer;
mod atlas;
mod config;
mod geo;
mod io;
mod model;
mod probes;
mod vps;
mod warts;

use anyhow::{Result, bail};
use clap::builder::RangedU64ValueParser;
use clap::{ArgAction, ArgGroup, ArgMatches, Command, arg, value_parser};
use indicatif::ParallelProgressIterator;
use polars::prelude::*;
use rayon::prelude::*;
use rstar::RTree;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use analyzer::AnycastAnalyzer;
use atlas::{
    AtlasClient, MeasureOptions, ProbeChoice, fetch_atlas_measurement, parse_atlas_id,
    run_measurement,
};
use io::{
    EMBEDDED_AIRPORTS, EMBEDDED_CITIES, decompress_gz, load_airports, load_input_data, progress_bar,
};
use model::{Airport, Disc, OutputRecord};
use probes::parse_probe_list;
use vps::VpTable;
use warts::load_warts_data;

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<()> {
    // Parse the command-line arguments
    let matches = parse_cmd();

    let input = matches.get_one::<PathBuf>("input");
    let output = matches.get_one::<PathBuf>("output");
    let atlas = matches.get_one::<String>("atlas");
    let vps = matches.get_one::<PathBuf>("vps");
    let warts: Option<Vec<PathBuf>> = matches
        .get_many::<PathBuf>("warts")
        .map(|paths| paths.cloned().collect());
    let measure: Option<Vec<String>> = matches
        .get_many::<String>("measure")
        .map(|targets| targets.cloned().collect());

    let dataset = matches.get_one::<String>("dataset").unwrap();
    let min_pop = *matches.get_one::<u32>("min_pop").unwrap();
    let pop_ratio = *matches.get_one::<f32>("pop_ratio").unwrap();
    let alpha = *matches.get_one::<f32>("alpha").unwrap();
    let threshold = *matches.get_one::<u32>("threshold").unwrap();
    let is_anycast = matches.get_flag("anycast");
    let is_accuracy = matches.get_flag("accuracy");
    let is_dry_run = matches.get_flag("dry_run");
    let api_key = matches.get_one::<String>("api_key");

    let is_save_key = matches.get_flag("save_api_key");
    if is_save_key {
        let path = config::save_key(api_key.map(String::as_str).unwrap_or_default())?;
        println!("Stored the RIPE Atlas API key in {}.", path.display());
    }

    let has_source = input.is_some() || atlas.is_some() || warts.is_some() || measure.is_some();
    if !has_source {
        if is_save_key {
            return Ok(());
        }
        bail!(
            "No input source given: pass one of --input, --atlas, --warts or --measure (see --help)."
        );
    }

    // Get optional RIPE Atlas ID
    let atlas_id = match atlas {
        Some(atlas_input) => Some(parse_atlas_id(atlas_input)?),
        None => None,
    };

    // Reject a run with nowhere to write before spending any time on it.
    if output.is_none() && atlas_id.is_none() && measure.is_none() {
        bail!("--output is required when using --input or --warts.");
    }

    // A new RIPE Atlas measurement runs first
    let measured = match measure {
        Some(ref targets) => {
            // A dry run only reads public data, so it needs no key.
            let key = if is_dry_run {
                None
            } else {
                Some(config::resolve_key(api_key.map(String::as_str))?)
            };
            let client = AtlasClient::new(key)?;
            let options = MeasureOptions {
                targets: targets.clone(),
                probes: match matches.get_one::<String>("probes") {
                    Some(list) => ProbeChoice::Explicit(parse_probe_list(list)?),
                    None => ProbeChoice::Spread(*matches.get_one::<usize>("num_probes").unwrap()),
                },
                packets: *matches.get_one::<u8>("packets").unwrap(),
                timeout: Duration::from_secs(
                    *matches.get_one::<u64>("measurement_timeout").unwrap(),
                ),
                validate: matches.get_flag("validate_probes"),
                dry_run: is_dry_run,
            };
            match run_measurement(&client, &options, threshold)? {
                Some(result) => Some(result),
                // --dry_run reported the probe selection and scheduled nothing.
                None => return Ok(()),
            }
        }
        None => None,
    };
    let (measured_ids, measured_df) = match measured {
        Some((ids, df)) => (Some(ids), Some(df)),
        None => (None, None),
    };

    // Load in the airports/cities (optionally filtered by minimum population)
    let airports = match dataset.as_str() {
        "airports" => {
            println!("Using embedded airports dataset.");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_AIRPORTS)?), 0)?
        }
        "cities" => {
            let filters: Vec<String> = [
                (min_pop > 0).then(|| format!("min pop: {}", min_pop)),
                (pop_ratio > 0.0).then(|| format!("relative: {}×max", pop_ratio)),
            ]
            .into_iter()
            .flatten()
            .collect();

            if filters.is_empty() {
                println!("Using embedded cities dataset.");
            } else {
                println!("Using embedded cities dataset ({}).", filters.join(", "));
            }
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES)?), min_pop)?
        }
        custom_path => {
            println!("Loading custom dataset from: {}", custom_path);
            load_airports(File::open(custom_path)?, min_pop)?
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

    // Load the vantage point coordinates, if one was given
    let vp_table = match vps {
        Some(path) => {
            println!("Loading vantage points from: {:?}", path);
            let table = VpTable::load(path)?;
            if table.len() == 0 {
                bail!("No usable vantage points in {:?}.", path);
            }
            let notes: Vec<String> = [
                (table.duplicates > 0).then(|| format!("{} duplicate", table.duplicates)),
                (table.malformed > 0).then(|| format!("{} malformed", table.malformed)),
            ]
            .into_iter()
            .flatten()
            .collect();
            if notes.is_empty() {
                println!("Loaded {} vantage points.", table.len());
            } else {
                println!(
                    "Loaded {} vantage points ({} line(s) skipped).",
                    table.len(),
                    notes.join(", ")
                );
            }
            Some(table)
        }
        None => None,
    };

    // Load input data (CSV file, warts files, or RIPE Atlas measurement)
    let in_df = if let Some(input_path) = input {
        println!("Loading input data from: {:?}", input_path);
        load_input_data(input_path, threshold, vp_table.as_ref())?
    } else if let Some(ref warts_paths) = warts {
        // --warts always uses a VPs file
        load_warts_data(warts_paths, vp_table.as_ref().unwrap(), threshold)?
    } else if let Some(df) = measured_df {
        df
    } else {
        fetch_atlas_measurement(atlas_id.unwrap(), threshold)?
    };

    // Runs that fetched or scheduled a measurement name their output after it.
    let output_path = match output {
        Some(p) => p.clone(),
        None => {
            let id =
                atlas_id.or_else(|| measured_ids.as_ref().and_then(|ids| ids.first().copied()));
            match id {
                Some(id) => PathBuf::from(format!("atlas_{}.csv", id)),
                None => bail!("--output is required when using --input or --warts."),
            }
        }
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
    let pb = progress_bar(num_targets as u64)?;

    // Perform geolocation
    let results: Vec<OutputRecord> = (0..num_targets)
        .map(|i| group_indices.get_as_series(i)) // Iterate over each series of indices
        .par_bridge() // Bridge to Rayon's multi-thread processing
        .progress_with(pb) // Progress bar
        .filter_map(|opt_indices_series| {
            opt_indices_series.map(|indices_series| {
                // Perform for each individual indices series
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
                    alpha,
                    pop_ratio,
                    is_anycast,
                    is_accuracy,
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
        let mut output_df = DataFrame::new(
            num_results,
            vec![
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
            ],
        )?;

        // Append accuracy columns if --accuracy flag is set
        if is_accuracy {
            let diameter_col = Series::new(
                "candidate_diameter".into(),
                results
                    .iter()
                    .map(|r| r.candidate_diameter.unwrap_or(0.0))
                    .collect::<Vec<f32>>(),
            );
            let constraints_col = Series::new(
                "num_constraints".into(),
                results
                    .iter()
                    .map(|r| r.num_constraints.unwrap_or(0))
                    .collect::<Vec<u32>>(),
            );
            output_df.with_column(diameter_col.into())?;
            output_df.with_column(constraints_col.into())?;
        }

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

/// Parse the command-line arguments.
///
/// Exactly one input source is used: `--input`, `--atlas`, `--warts` or `--measure`.
/// The `measurement` group holds everything that configures a live measurement, so
/// those options are rejected without `--measure` and alongside another source.
fn parse_cmd() -> ArgMatches {
    Command::new("migreedy")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Remi Hendriks <remi.hendriks@utwente.nl>")
        .about("A fast, parallel improved version of the iGreedy algorithm for large-scale anycast-aware geolocation")
        .arg(arg!(-i --input <FILE> "Input CSV file: addr,hostname,lat,lon,rtt (or addr,hostname,rtt with --vps)")
            .value_parser(value_parser!(PathBuf)))
        .arg(arg!(--atlas <ID> "RIPE Atlas measurement ID or URL"))
        .arg(arg!(--warts <PATH> "scamper warts files to read (.warts/.warts.gz); accepts files, globs and directories. Requires --vps")
            .value_parser(value_parser!(PathBuf))
            .num_args(1..)
            .requires("vps"))
        .group(ArgGroup::new("source").args(["input", "atlas", "warts"]))
        .arg(arg!(--measure <TARGET> "Schedule RIPE Atlas ping measurements to these targets and geolocate the results (needs an API key)")
            .num_args(1..)
            .conflicts_with("source"))
        .arg(arg!(-o --output <PATH> "Path to write output (defaults to atlas_<ID>.csv with --atlas and --measure)")
            .value_parser(value_parser!(PathBuf)))
        .arg(arg!(--vps <FILE> "Vantage point coordinates file: whitespace-separated 'hostname lat lon', no header")
            .value_parser(value_parser!(PathBuf))
            .conflicts_with_all(["atlas", "measure"]))
        .arg(arg!(-d --dataset <NAME> "Dataset to use: 'cities' (embedded), 'airports' (embedded), or path to custom CSV")
            .default_value("cities"))
        .arg(arg!(-m --min_pop <N> "Absolute minimum population threshold (filter cities at load time)")
            .value_parser(value_parser!(u32))
            .default_value("0"))
        .arg(arg!(-p --pop_ratio <RATIO> "Relative population threshold (0.0-1.0): keep cities with pop >= max_pop * ratio within each geolocation")
            .value_parser(value_parser!(f32))
            .default_value("0.0"))
        .arg(arg!(-a --alpha <ALPHA> "Alpha (population vs distance score tuning)")
            .value_parser(value_parser!(f32))
            .default_value("1.0"))
        .arg(arg!(-t --threshold <MS> "Discard disks with RTT > threshold")
            .value_parser(value_parser!(u32))
            .default_value("0"))
        .arg(arg!(--anycast "Only output anycast geolocations (skip unicast)").action(ArgAction::SetTrue))
        .arg(arg!(--accuracy "Include accuracy metrics: candidate_diameter (km) and num_constraints").action(ArgAction::SetTrue))
        .arg(arg!(--api_key <KEY> "RIPE Atlas API key with measurement creation permission (default: $MIGREEDY_ATLAS_KEY, else the stored key)"))
        .arg(arg!(--save_api_key "Store --api_key for later runs; can be used on its own to configure the key")
            .action(ArgAction::SetTrue)
            .requires("api_key"))
        .arg(arg!(--probes <IDS> "Probes to measure from: comma-separated IDs, or a file listing them")
            .conflicts_with("num_probes"))
        .arg(arg!(--num_probes <N> "Number of probes to select, spread for the widest global coverage")
            .value_parser(RangedU64ValueParser::<usize>::new().range(1..))
            .default_value("100"))
        .arg(arg!(--packets <N> "Ping packets sent per probe")
            .value_parser(RangedU64ValueParser::<u8>::new().range(1..))
            .default_value("1"))
        .arg(arg!(--measurement_timeout <SECONDS> "How long to wait for measurement results before using what has arrived")
            .value_parser(value_parser!(u64))
            .default_value("300"))
        .arg(arg!(--validate_probes "Also ping RIPE Atlas anchors to drop probes whose location the RTTs rule out (costs extra credits)")
            .action(ArgAction::SetTrue))
        .arg(arg!(--dry_run "Select and report probes without scheduling anything").action(ArgAction::SetTrue))
        .group(ArgGroup::new("measurement")
            .multiple(true)
            .args(["probes", "num_probes", "packets", "measurement_timeout", "validate_probes", "dry_run"])
            .requires("measure")
            .conflicts_with("source"))
        .get_matches()
}
