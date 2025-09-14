use anyhow::Result;
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use std::path::PathBuf;
use std::fs::File;

// Constants TODO verify these values
const FIBER_RI: f32 = 1.52;
const SPEED_OF_LIGHT: f32 = 299792.458; // km/s
const EARTH_RADIUS_KM: f32 = 6371.0;

// Data Structures
/// Represents a single airport from the airports.csv dataset
/// Fields:
/// * iata: IATA airport code (String)
/// * lat: Latitude in degrees (f32)
/// * lon: Longitude in degrees (f32)
/// * pop: Population (u32)
/// * city: City name (String)
/// * country_code: Country code (String)
/// * lat_rad: Latitude in radians (f32)
/// * lon_rad: Longitude in radians (f32)
#[derive(Debug, Clone)]
struct Airport {
    iata: String,
    lat: f32, // TODO store as f32 for memory efficiency
    lon: f32,
    pop: u32,
    city: String, // TODO not used?
    country_code: String, // ISO 3166-1 alpha-2 (always 2 chars) TODO consider using [u8; 2]?
    lat_rad: f32, // TODO store as f32 for memory efficiency
    lon_rad: f32,
}

/// Represents a single vantage point's disc
/// Fields:
/// * target: Target IP address (String)
/// * hostname: Vantage point hostname (String)
/// * vp_lat: Vantage point latitude in degrees (f32)
/// * vp_lon: Vantage point longitude in degrees (f32)
/// * rtt: Measured RTT in milliseconds (f32)
/// * lat_rad: Vantage point latitude in radians (f32)
/// * lon_rad: Vantage point longitude in radians (f32)
/// * radius: Calculated disc radius in kilometers (f32)
/// * processed: Whether this disc has been processed (bool)
/// * original_index: Original index in the input data (usize)
#[derive(Debug, Clone)]
struct Disc {
    // Original data
    target: String, // TODO consider using IpAddr type? or either u32/u128 for IPv4/IPv6?
    hostname: String,
    vp_lat: f32, // TODO store immediately as rad?
    vp_lon: f32,
    rtt: f32, // TODO only store radius?
    // Calculated fields
    lat_rad: f32,
    lon_rad: f32,
    radius: f32,
    // State for the algorithm
    processed: bool,
    // Original index to keep track
    original_index: usize, // TODO needed?
}

/// Represents a single output record after geolocation
/// Fields:
/// * target: Target IP address (String)
/// * vp: Vantage point hostname (String)
/// * vp_lat: Vantage point latitude in degrees (f32)
/// * vp_lon: Vantage point longitude in degrees (f32)
/// * radius: Calculated disc radius in kilometers (f32)
/// * pop_iata: Geolocated airport IATA code (String)
/// * pop_lat: Geolocated airport latitude in degrees (f32)
/// * pop_lon: Geolocated airport longitude in degrees (f32)
/// * pop_city: Geolocated airport city name (String)
/// * pop_cc: Geolocated airport country code (String)
#[derive(Debug, Default)]
struct OutputRecord {
    target: String, // TODO consider using IpAddr type? or either u32/u128 for IPv4/IPv6?
    vp: String,
    vp_lat: f32,
    vp_lon: f32,
    radius: f32,
    pop_iata: String,
    pop_lat: f32,
    pop_lon: f32,
    pop_city: String,
    pop_cc: String, // TODO consider using [u8; 2]?
}

/// Haversine formula to calculate the great-circle distance between two points
/// given their latitude and longitude in radians.
/// Returns distance in kilometers.
/// Parameters:
/// * lat1: Latitude of point 1 in radians (f32)
/// * lon1: Longitude of point 1 in radians (f32)
/// * lat2: Latitude of point 2 in radians (f32)
/// * lon2: Longitude of point 2 in radians (f32)
/// Returns:
/// * Distance in kilometers (f32)
fn haversine_distance(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_KM * c
}


/// The main analyzer struct that encapsulates the anycast analysis logic.
/// Fields:
/// * alpha: Weighting factor for population vs distance in geolocation (f32)
/// * airports: Reference to the list of airports (slice of Airport)
/// * all_discs: List of all discs (i.e., latency measurements) for the current target (Vec of Disc)
struct AnycastAnalyzer<'a> {
    alpha: f32,
    airports: &'a [Airport],
    all_discs: Vec<Disc>,
}

impl<'a> AnycastAnalyzer<'a> {
    /// Constructor for the analyzer.
    /// Parameters:
    /// * discs: List of discs for the current target (Vec of Disc)
    /// * airports: Reference to the list of airports (slice of Airport)
    /// * alpha: Weighting factor for population vs distance in geolocation (f32)
    /// Returns:
    /// * Instance of AnycastAnalyzer
    /// Note: Discs are sorted by RTT in ascending order.
    fn new(mut discs: Vec<Disc>, airports: &'a [Airport], alpha: f32) -> Self {
        // Sort by RTT, such that lower RTT discs are processed first
        discs.sort_unstable_by(|a, b| a.rtt.partial_cmp(&b.rtt).unwrap());
        Self {
            alpha,
            airports,
            all_discs: discs,
        }
    }

    /// Analyze the discs (belonging to a single target) to find anycast sites.
    /// Returns:
    /// * Option<Vec<OutputRecord>>: List of geolocated sites, or None if no anycast found
    fn analyze(mut self) -> Option<Vec<OutputRecord>> {
        let mut results = Vec::new();
        let mut chosen_airports = std::collections::HashSet::new();
        let target_ip = self.all_discs.first()?.target.clone();

        loop {
            // Find number of anycast sites (maximum independent set of discs)
            let (num_sites, mis_indices) = self.enumeration();
            if num_sites <= 1 {
                break; // Unicast or not enough data
            }

            let mut iteration_found_new_site = false;

            // Geolocate each anycast site (disc in MIS)
            for disc_index in mis_indices {
                // Check if this disc has already been processed in a previous iteration
                if self.all_discs[disc_index].processed {
                    continue;
                }
                let geolocation_result = self.geolocation(&self.all_discs[disc_index]);

                let disc_in_mis = &self.all_discs[disc_index];


                if let Some(best_airport) = geolocation_result {
                    if chosen_airports.contains(&best_airport.iata) {
                        // Airport already found by a lower-RTT disc, so mark this one as processed and skip
                        self.all_discs[disc_index].processed = true;
                        continue;
                    }
                    chosen_airports.insert(best_airport.iata.clone());

                    results.push(OutputRecord {
                        target: target_ip.clone(),
                        vp: disc_in_mis.hostname.clone(),
                        vp_lat: disc_in_mis.vp_lat,
                        vp_lon: disc_in_mis.vp_lon,
                        radius: disc_in_mis.radius,
                        pop_iata: best_airport.iata.clone(),
                        pop_lat: best_airport.lat,
                        pop_lon: best_airport.lon,
                        pop_city: best_airport.city.clone(),
                        pop_cc: best_airport.country_code.clone(),
                    });

                    // Modify the "master" list of discs for the next enumeration run
                    let master_disc = &mut self.all_discs[disc_index];
                    master_disc.lat_rad = best_airport.lat_rad;
                    master_disc.lon_rad = best_airport.lon_rad;
                    master_disc.radius = 0.1; // Geolocation radius
                    master_disc.processed = true;

                    iteration_found_new_site = true;
                    break; // Re-run enumeration
                } else {
                    // Geolocation failed for this disc
                    let failed_disc = &self.all_discs[disc_index];

                    results.push(OutputRecord {
                        target: target_ip.clone(),
                        vp: failed_disc.hostname.clone(),
                        vp_lat: failed_disc.vp_lat,
                        vp_lon: failed_disc.vp_lon,
                        radius: failed_disc.radius,
                        pop_iata: "NoCity".to_string(),
                        pop_lat: failed_disc.vp_lat,
                        pop_lon: failed_disc.vp_lon,
                        pop_city: "N/A".to_string(),
                        pop_cc: "N/A".to_string(),
                    });
                    self.all_discs[disc_index].processed = true;
                }
            }

            if !iteration_found_new_site {
                break; // No new sites found, exit the loop
            }
        }

        if results.is_empty() {
            None
        } else {
            Some(results)
        }
    }


    /// Finds the maximum independent set of discs (non-overlapping discs).
    /// Each disc represents an anycast site.
    /// Returns:
    /// * (usize, Vec<usize>): Number of discs (anycast sites) and the list of indices of discs in the MIS
    fn enumeration(&self) -> (usize, Vec<usize>) {
        let mut mis_indices: Vec<usize> = Vec::new();

        for candidate in &self.all_discs {
            let is_overlapping = mis_indices.iter().any(|&existing_index| {
                // Get a temporary reference to the disc already in the MIS
                let existing_disc = &self.all_discs[existing_index];

                let distance = haversine_distance(
                    candidate.lat_rad,
                    candidate.lon_rad,
                    existing_disc.lat_rad,
                    existing_disc.lon_rad,
                );
                distance <= candidate.radius + existing_disc.radius
            });

            if !is_overlapping {
                // Store the index of the disc in the MIS
                mis_indices.push(candidate.original_index);
            }
        }
        (mis_indices.len(), mis_indices)
    }

    /// Geolocates a disc by finding the best matching airport within its radius.
    /// The best airport is determined by a scoring function that balances population and distance.
    /// Parameters:
    /// * disc: Reference to the disc to geolocate (&Disc)
    /// Returns:
    /// * Option<Airport>: The best matching airport, or None if no airport found within the disc
    fn geolocation(&self, disc: &Disc) -> Option<Airport> {
        let center_lat = disc.lat_rad;
        let center_lon = disc.lon_rad;

        // Bounding box pre-filter for performance
        let delta_lat = disc.radius / EARTH_RADIUS_KM;
        let min_lat = center_lat - delta_lat;
        let max_lat = center_lat + delta_lat;

        let delta_lon = disc.radius / (EARTH_RADIUS_KM * center_lat.cos());
        let min_lon = center_lon - delta_lon;
        let max_lon = center_lon + delta_lon;

        // First, filter airports within the bounding box, then calculate exact distances
        // and filter those within the disc radius
        let airports_inside_disk: Vec<_> = self.airports
            .iter()
            .filter(|a| {
                a.lat_rad >= min_lat && a.lat_rad <= max_lat && a.lon_rad >= min_lon && a.lon_rad <= max_lon
            })
            .map(|a| {
                let dist = haversine_distance(center_lat, center_lon, a.lat_rad, a.lon_rad);
                (a, dist)
            })
            .filter(|(_a, dist)| *dist <= disc.radius)
            .collect();

        if airports_inside_disk.is_empty() {
            return None;
        }

        let total_pop: f32 = airports_inside_disk.iter().map(|(a, _)| a.pop as f32).sum();
        let total_dist: f32 = airports_inside_disk.iter().map(|(_, d)| *d).sum();

        // Find airport with the highest score
        let best_airport = airports_inside_disk.into_iter()
            .max_by(|(a1, d1), (a2, d2)| {
                let pop_score1 = if total_pop > 0.0 { a1.pop as f32 / total_pop } else { 0.0 };
                let dist_score1 = if total_dist > 0.0 { d1 / total_dist } else { 0.0 };
                let score1 = self.alpha * pop_score1 - (1.0 - self.alpha) * dist_score1;

                let pop_score2 = if total_pop > 0.0 { a2.pop as f32 / total_pop } else { 0.0 };
                let dist_score2 = if total_dist > 0.0 { d2 / total_dist } else { 0.0 };
                let score2 = self.alpha * pop_score2 - (1.0 - self.alpha) * dist_score2;

                score1.partial_cmp(&score2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(a, _)| a.clone());

        best_airport
    }
}


/// Parse command-line arguments
/// Fields:
/// * input: Input CSV file path (PathBuf)
/// * output: Output CSV file path (PathBuf)
/// * airports: Path to airports dataset (PathBuf)
/// * alpha: Optional weighting factor for population vs distance in geolocation (f32)
/// * threshold: Optional RTT threshold to filter discs (f32)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "Input CSV file")]
    input: PathBuf,

    #[arg(short, long, help = "Output CSV file")]
    output: PathBuf,

    #[arg(long, default_value = "../datasets/airports.csv", help="Path to airports dataset")]
    airports: PathBuf,

    #[arg(short, long, default_value_t = 1.0, help = "Alpha (population vs distance score tuning)")]
    alpha: f32,

    #[arg(short, long, default_value_t = 0, help = "Discard disks with RTT > threshold")]
    threshold: u32,
}

/// Load and preprocess airports data from the given path.
/// The airports data is expected to be in a tab-separated format without a header.
/// Expected columns: iata, size, name, lat, lon, country_code, city, pop
/// Additional fields lat_rad and lon_rad are computed in radians.
/// Parameters:
/// * path: Path to the airports CSV file (PathBuf)
/// Returns:
/// * Result<Vec<Airport>>: List of airports, or error if loading fails
fn load_airports(path: &PathBuf) -> Result<Vec<Airport>> { // TODO
    // Define the schema for the airports data
    let airports_schema = Arc::new(Schema::from_iter([
        Field::new(PlSmallStr::from("iata"), DataType::String),
        Field::new(PlSmallStr::from("size"), DataType::String),
        Field::new(PlSmallStr::from("name"), DataType::String),
        Field::new(PlSmallStr::from("lat"), DataType::String),
        Field::new(PlSmallStr::from("lon"), DataType::String),
        Field::new(PlSmallStr::from("country_code"), DataType::String),
        Field::new(PlSmallStr::from("city"), DataType::String),
        Field::new(PlSmallStr::from("city"), DataType::String),
        Field::new(PlSmallStr::from("pop"), DataType::UInt32),
        Field::new(PlSmallStr::from("heuristic"), DataType::String),
        // TODO clean-up file and add/remove fields as necessary
    ]));

    // Specify CSV read options
    let airports_read_options = CsvReadOptions {
        has_header: false,
        schema: Some(airports_schema),
        parse_options: Arc::new(
            CsvParseOptions::default()
                .with_separator(b'\t')
        ),
        ..Default::default()
    };

    let airports_file = File::open(path)?;
    let airports_df = CsvReader::new(airports_file)
        .with_options(airports_read_options)
        .finish()?;

    // TODO compute lat_rad and lon_rad

    // Convert DataFrame to a Vec of structs
    let iata = airports_df.column("iata")?.str()?;
    let lat = airports_df.column("lat")?.f32()?;
    let lon = airports_df.column("lon")?.f32()?;
    let pop = airports_df.column("pop")?.u32()?;
    let city = airports_df.column("city")?.str()?;
    let country_code = airports_df.column("country_code")?.str()?;
    let lat_rad = airports_df.column("lat_rad")?.f32()?;
    let lon_rad = airports_df.column("lon_rad")?.f32()?;
    // TODO why not load directly as vector of structs?
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

/// Load and preprocess input data from the given path.
/// The input data is expected to be in a CSV format with a header.
/// Expected columns: target, hostname, lat, lon, rtt
/// Additional fields lat_rad, lon_rad, and radius are computed.
/// Parameters:
/// * path: Path to the input CSV file (PathBuf)
/// * threshold: RTT threshold to filter discs (u32)
/// Returns:
/// * Result<Vec<DataFrame>>: List of DataFrames, each corresponding to a group of rows for a specific target IP
fn load_input_data(path: &PathBuf, threshold: u32) -> Result<Vec<DataFrame>> { // TODO
    // TODO dataframe needed? or just load directly as Vec<Disc>?
    // Define input schema and read options
    let input_columns = Arc::from([
        PlSmallStr::from("target"),
        PlSmallStr::from("hostname"),
        PlSmallStr::from("lat"),
        PlSmallStr::from("lon"),
        PlSmallStr::from("rtt"),
    ]);

    let read_options = CsvReadOptions {
        has_header: false,
        skip_rows: 1,
        columns: Some(input_columns),
        ..Default::default()
    };

    let input_file = File::open(path)?;
    let mut in_df = CsvReader::new(input_file)
        .with_options(read_options)
        .finish()?;

    // // Apply RTT threshold if provided TODO
    // if threshold > 0 {
    //     in_df = in_df.filter(col("rtt").lt_eq(lit(threshold)));
    //     println!("Applied RTT threshold filter: {} ms. Records after filtering: {}", threshold, in_df.height());
    // }

    // Add calculated fields TODO
    // let lat_rad = in_df.column("lat")?.f32()?.apply(|v| v.to_radians()).into_series();
    // let lon_rad = in_df.column("lon")?.f32()?.apply(|v| v.to_radians()).into_series();
    // let radius = in_df.column("rtt")?.f32()?.apply(|rtt| {
    //     (rtt.unwrap_or(0.0) as f32 * 0.001 * SPEED_OF_LIGHT / FIBER_RI / 2.0)
    // }).into_series();
    //
    // in_df.with_column(lat_rad.with_name("lat_rad"))?;
    // in_df.with_column(lon_rad.with_name("lon_rad"))?;
    // in_df.with_column(radius.with_name("radius"))?;

    // Group by target IP
    let groups_df = in_df.group_by(["target"])?.groups()?;
    // Extract the "groups" column which contains the indices for each group
    let indices = groups_df.column("groups")?.list()?;
    // Create a Vec<DataFrame> where each DataFrame corresponds to a group of rows for a specific target IP
    let groups: Vec<DataFrame> = indices
        .into_iter()
        .filter_map(|opt_indices_series| {
            opt_indices_series.map(|indices_series| {
                let indices_ca = indices_series.u32()?;
                in_df.take(indices_ca)
            })
        })
        .map(|result| result.map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;

    Ok(groups)
}

/// Main function
/// Steps:
/// * Parse command-line arguments
/// * Load and preprocess airports data
/// * Load and preprocess input data
/// * Apply RTT threshold filter if specified
/// * Calculate additional fields (lat_rad, lon_rad, radius)
/// * Group input data by target IP
/// * Process each group in parallel using Rayon
/// * For each group, create an AnycastAnalyzer instance and run iGreedy algorithm
/// * Collect results and write to output CSV file
fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading airports data from: {:?}", args.airports);
    let airports = load_airports(&args.airports)?;
    println!("Loaded {} airports.", airports.len());

    // Load and preprocess input data (latency measurements)
    println!("Loading input data from: {:?}", args.input);
    let groups = load_input_data(&args.input, args.threshold)?;

    let num_targets = groups.len();
    println!("Starting parallel processing for {} targets...", num_targets);

    // Setup progress bar TODO test
    let pb = ProgressBar::new(num_targets as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    // Process each group in parallel using Rayon
    // Collect results into a single Vec<OutputRecord>
    let results: Vec<OutputRecord> = groups
        .into_par_iter() // Parallel iterator from Rayon
        .progress_with(pb)
        .filter_map(|group_df| {
            // Extract columns as Series
            let target = group_df.column("target").unwrap().str().unwrap();
            let hostname = group_df.column("hostname").unwrap().str().unwrap();
            let vp_lat = group_df.column("lat").unwrap().f32().unwrap();
            let vp_lon = group_df.column("lon").unwrap().f32().unwrap();
            let rtt = group_df.column("rtt").unwrap().f32().unwrap();
            let lat_rad = group_df.column("lat_rad").unwrap().f32().unwrap();
            let lon_rad = group_df.column("lon_rad").unwrap().f32().unwrap();
            let radius = group_df.column("radius").unwrap().f32().unwrap();

            // Create Vec<Disc> for the current group
            let discs: Vec<Disc> = (0..group_df.height())
                .map(|i| Disc {
                    target: target.get(i).unwrap_or("").to_string(),
                    hostname: hostname.get(i).unwrap_or("").to_string(),
                    vp_lat: vp_lat.get(i).unwrap_or(0.0),
                    vp_lon: vp_lon.get(i).unwrap_or(0.0),
                    rtt: rtt.get(i).unwrap_or(0.0),
                    lat_rad: lat_rad.get(i).unwrap_or(0.0),
                    lon_rad: lon_rad.get(i).unwrap_or(0.0),
                    radius: radius.get(i).unwrap_or(0.0),
                    processed: false,
                    original_index: i,
                })
                .collect();

            // Create analyzer instance and run iGreedy algorithm
            let analyzer = AnycastAnalyzer::new(discs, &airports, args.alpha);
            analyzer.analyze()
        })
        .flatten() // Flatten the Option<Vec<OutputRecord>> into one big Vec
        .collect();

    println!("Analysis complete. Found {} geolocated sites.", results.len());

    if !results.is_empty() {
        println!("Saving results to {:?}...", args.output);
        let mut output_df = DataFrame::new(vec![
            Series::new("target".into(), results.iter().map(|r| r.target.as_str()).collect::<Vec<_>>()).into(),
            Series::new("vp".into(), results.iter().map(|r| r.vp.as_str()).collect::<Vec<_>>()).into(),

            Series::new("vp_lat".into(), results.iter().map(|r| r.vp_lat).collect::<Vec<_>>()).into(),
            Series::new("vp_lon".into(), results.iter().map(|r| r.vp_lon).collect::<Vec<_>>()).into(),
            Series::new("radius".into(), results.iter().map(|r| r.radius).collect::<Vec<_>>()).into(),

            Series::new("pop_iata".into(), results.iter().map(|r| r.pop_iata.as_str()).collect::<Vec<_>>()).into(),
            Series::new("pop_lat".into(), results.iter().map(|r| r.pop_lat).collect::<Vec<_>>()).into(),
            Series::new("pop_lon".into(), results.iter().map(|r| r.pop_lon).collect::<Vec<_>>()).into(),
            Series::new("pop_city".into(), results.iter().map(|r| r.pop_city.as_str()).collect::<Vec<_>>()).into(),
            Series::new("pop_cc".into(), results.iter().map(|r| r.pop_cc.as_str()).collect::<Vec<_>>()).into(),
        ])?;

        let mut file = File::create(args.output)?;
        CsvWriter::new(&mut file)
            .with_separator(b'\t')
            .finish(&mut output_df)?;

        println!("Results successfully saved.");
    } else {
        println!("No anycast instances found, no output file written.");
    }

    Ok(())
}