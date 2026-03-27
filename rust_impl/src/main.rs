use anyhow::{bail, Result};
use clap::Parser;
use flate2::read::GzDecoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::PathBuf;

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Embedded datasets (gzip-compressed) at compile time so the binary is self-contained.
static EMBEDDED_AIRPORTS: &[u8] = include_bytes!("../../datasets/airports.csv.gz");
static EMBEDDED_CITIES500: &[u8] = include_bytes!("../../datasets/cities500.csv.gz");
static EMBEDDED_CITIES1_000: &[u8] = include_bytes!("../../datasets/cities1_000.csv.gz");
static EMBEDDED_CITIES5_000: &[u8] = include_bytes!("../../datasets/cities5_000.csv.gz");
static EMBEDDED_CITIES15_000: &[u8] = include_bytes!("../../datasets/cities15_000.csv.gz");
static EMBEDDED_CITIES100_000: &[u8] = include_bytes!("../../datasets/cities100_000.csv.gz");


/// Decompress a gzip-compressed byte slice into a Vec<u8>.
fn decompress_gz(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

// Constants
const FIBER_RI: f32 = 1.52;
const SPEED_OF_LIGHT: f32 = 299792.458; // km/s
const EARTH_RADIUS_KM: f32 = 6371.0;

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
    iata: String, // always 3 chars
    lat: f32,
    lon: f32,
    pop: u32,
    city: String,
    country_code: String, // ISO 3166-1 alpha-2 (always 2 chars)
    lat_rad: f32,
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
    anycast_only: bool,
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
    fn new(mut discs: Vec<Disc>, airports: &'a [Airport], alpha: f32, anycast_only: bool) -> Self {
        // Sort by RTT, such that lower RTT discs are processed first
        discs.sort_unstable_by(|a, b| a.rtt.partial_cmp(&b.rtt).unwrap());
        Self {
            alpha,
            airports,
            all_discs: discs,
            anycast_only,
        }
    }

    /// Analyze the discs (belonging to a single target) to find anycast sites.
    /// Returns:
    /// * Option<Vec<OutputRecord>>: List of geolocated sites, or None if no anycast found
    fn analyze(self) -> Vec<OutputRecord> {
        let target_ip = self.all_discs.first().unwrap().target.clone();

        // get the maximum independent set of discs (non-overlapping discs)
        let (num_sites, mis_indices) = self.enumeration();

        // skip targets with no discs, or unicast when --anycast is set
        if num_sites == 0 || (self.anycast_only && num_sites <= 1) {
            return vec![];
        }

        let mut results = Vec::new();
        let mut chosen_airports = std::collections::HashSet::new();

        // Geolocate each anycast site from the single MIS result (in order of increasing RTT).
        for disc_index in mis_indices {
            let disc_in_mis = &self.all_discs[disc_index];
            let cluster = self.build_cluster(disc_in_mis);
            let geolocation_result = self.geolocation(&cluster);

            if let Some(best_airport) = geolocation_result {
                // If the best airport for this disc has already been assigned to a
                // lower-RTT disc (processed earlier in this loop), we skip this disc.
                if chosen_airports.contains(&best_airport.iata) {
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
            } else {
                // Geolocation failed for this disc (no airports found within its radius).
                results.push(OutputRecord {
                    target: target_ip.clone(),
                    vp: disc_in_mis.hostname.clone(),
                    vp_lat: disc_in_mis.vp_lat,
                    vp_lon: disc_in_mis.vp_lon,
                    radius: disc_in_mis.radius,
                    pop_iata: "NoCity".to_string(),
                    pop_lat: disc_in_mis.vp_lat, // Fallback to the VP's own location
                    pop_lon: disc_in_mis.vp_lon,
                    pop_city: "N/A".to_string(),
                    pop_cc: "N/A".to_string(),
                });
            }
        }

        results
    }

    /// Finds the maximum independent set of discs (non-overlapping discs).
    /// Each disc represents an anycast site.
    /// Returns:
    /// * (usize, Vec<usize>): Number of discs (anycast sites) and the list of indices of discs in the MIS
    fn enumeration(&self) -> (usize, Vec<usize>) {
        let mut mis_indices: Vec<usize> = Vec::new();

        for (i, candidate) in self.all_discs.iter().enumerate() {
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
                mis_indices.push(i);
            }
        }
        (mis_indices.len(), mis_indices)
    }

    // TODO we should discard discs that overlap multiple MIS discs (we cannot know which MIS they reached)

    /// Builds the cluster for a given MIS disc: all discs whose centre is within
    /// the sum of their radii (i.e., they overlap with the MIS disc).
    /// These discs all likely measure the same anycast site.
    /// Parameters:
    /// * mis_disc: Reference to the MIS disc (&Disc)
    /// Returns:
    /// * Vec<&Disc>: References to all overlapping discs (always includes mis_disc itself)
    fn build_cluster<'s>(&'s self, mis_disc: &Disc) -> Vec<&'s Disc> {
        self.all_discs
            .iter()
            .filter(|d| {
                let dist = haversine_distance(
                    mis_disc.lat_rad, mis_disc.lon_rad,
                    d.lat_rad, d.lon_rad,
                );
                dist <= mis_disc.radius + d.radius
            })
            .collect()
    }

    /// Geolocates a site by finding the best matching airport within the intersection
    /// of all discs in the cluster (i.e., within the radius of every disc).
    /// The smallest disc anchors the bounding-box pre-filter and the distance scoring.
    ///
    /// If the full intersection contains no cities, progressively relaxes constraints:
    /// discs are sorted by radius (ascending) and intersected one by one. When adding
    /// the next disc would empty the candidate set, the previous (tightest non-empty)
    /// intersection is used instead. This preserves as many constraints as possible
    /// while guaranteeing a result when at least the smallest disc contains a city.
    ///
    /// Parameters:
    /// * cluster: Slice of disc references forming one site's cluster (&[&Disc])
    /// Returns:
    /// * Option<Airport>: The best matching airport, or None if no candidate is found
    fn geolocation(&self, cluster: &[&Disc]) -> Option<Airport> {
        // Smallest disc = tightest single constraint; used for bounding box and distance scoring
        let smallest = cluster
            .iter()
            .min_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap())?;

        let center_lat = smallest.lat_rad;
        let center_lon = smallest.lon_rad;

        // Bounding box pre-filter based on the smallest disc
        let delta_lat = smallest.radius / EARTH_RADIUS_KM;
        let min_lat = center_lat - delta_lat;
        let max_lat = center_lat + delta_lat;

        let delta_lon = smallest.radius / (EARTH_RADIUS_KM * center_lat.cos());
        let min_lon = center_lon - delta_lon;
        let max_lon = center_lon + delta_lon;

        // Collect airports in the bounding box, storing distance from the smallest disc's centre
        let airports_in_bbox: Vec<(&Airport, f32)> = self
            .airports
            .iter()
            .filter(|a| {
                a.lat_rad >= min_lat
                    && a.lat_rad <= max_lat
                    && a.lon_rad >= min_lon
                    && a.lon_rad <= max_lon
            })
            .map(|a| {
                let dist = haversine_distance(center_lat, center_lon, a.lat_rad, a.lon_rad);
                (a, dist)
            })
            .collect();

        if airports_in_bbox.is_empty() {
            return None;
        }

        // Sort cluster discs by radius (ascending) so we apply the tightest constraints first
        let mut sorted_cluster: Vec<&&Disc> = cluster.iter().collect();
        sorted_cluster.sort_unstable_by(|a, b| a.radius.partial_cmp(&b.radius).unwrap());

        // Progressively intersect: add one disc at a time (smallest first).
        // Keep the last non-empty candidate set as fallback.
        let mut candidates = airports_in_bbox;
        let mut last_valid = candidates.clone();

        for disc in &sorted_cluster {
            candidates.retain(|(a, _)| {
                let dist = haversine_distance(disc.lat_rad, disc.lon_rad, a.lat_rad, a.lon_rad);
                dist <= disc.radius
            });

            if candidates.is_empty() {
                // Adding this disc emptied the set — use the previous valid set
                break;
            }
            last_valid = candidates.clone();
        }

        if last_valid.is_empty() {
            return None;
        }

        let total_pop: f32 = last_valid.iter().map(|(a, _)| a.pop as f32).sum();
        let total_dist: f32 = last_valid.iter().map(|(_, d)| *d).sum();

        // Find airport with the highest score
        last_valid
            .into_iter()
            .max_by(|(a1, d1), (a2, d2)| {
                let pop_score1 = if total_pop > 0.0 { a1.pop as f32 / total_pop } else { 0.0 };
                let dist_score1 = if total_dist > 0.0 { d1 / total_dist } else { 0.0 };
                let score1 = self.alpha * pop_score1 - (1.0 - self.alpha) * dist_score1;

                let pop_score2 = if total_pop > 0.0 { a2.pop as f32 / total_pop } else { 0.0 };
                let dist_score2 = if total_dist > 0.0 { d2 / total_dist } else { 0.0 };
                let score2 = self.alpha * pop_score2 - (1.0 - self.alpha) * dist_score2;

                score1.partial_cmp(&score2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(a, _)| a.clone())
    }
}

/// Parse command-line arguments
/// Fields:
/// * input: Input CSV file path (PathBuf)
/// * output: Output CSV file path (PathBuf)
/// * atlas: RIPE Atlas measurement ID or URL
/// * airports: Path to airports dataset (PathBuf)
/// * alpha: Optional weighting factor for population vs distance in geolocation (f32)
/// * threshold: Optional RTT threshold to filter discs (f32)
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
        default_value = "cities500",
        help = "Dataset to use: cities500, cities1000, cities5000, cities15000, airports, or path to custom CSV"
    )]
    dataset: String,

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

// ── RIPE Atlas API types ─────────────────────────────────────────────

#[derive(Deserialize)]
struct AtlasResult {
    dst_addr: Option<String>,
    #[serde(default)]
    min: Option<f64>,
    prb_id: u32,
    #[serde(rename = "type")]
    measurement_type: Option<String>,
    // For traceroute: extract min RTT from last hop
    result: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ProbeGeometry {
    coordinates: Option<Vec<f64>>, // [lon, lat]
}

#[derive(Deserialize)]
struct ProbeInfo {
    id: u32,
    geometry: Option<ProbeGeometry>,
}

#[derive(Deserialize)]
struct ProbeResponse {
    results: Vec<ProbeInfo>,
    next: Option<String>,
}

/// Extract the measurement ID from either a plain numeric string or a RIPE Atlas URL.
/// Supported URL formats:
/// - https://atlas.ripe.net/measurements/12345/
/// - https://atlas.ripe.net/api/v2/measurements/12345/
/// - https://atlas.ripe.net/api/v2/measurements/12345/results/
fn parse_atlas_id(input: &str) -> Result<u64> {
    // Try plain number first
    if let Ok(id) = input.trim().parse::<u64>() {
        return Ok(id);
    }

    // Try to extract from URL
    let parts: Vec<&str> = input.trim_end_matches('/').split('/').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "measurements" {
            if let Some(id_str) = parts.get(i + 1) {
                if let Ok(id) = id_str.parse::<u64>() {
                    return Ok(id);
                }
            }
        }
    }

    bail!("Could not parse RIPE Atlas measurement ID from: {}", input)
}

/// Extract minimum RTT from a traceroute result by finding the last hop with valid RTT.
fn extract_traceroute_min_rtt(result_value: &serde_json::Value) -> Option<f64> {
    let hops = result_value.as_array()?;
    // Iterate hops in reverse to find the last one with a valid RTT
    for hop in hops.iter().rev() {
        if let Some(results) = hop.get("result").and_then(|r| r.as_array()) {
            let rtts: Vec<f64> = results
                .iter()
                .filter_map(|r| r.get("rtt").and_then(|v| v.as_f64()))
                .collect();
            if !rtts.is_empty() {
                return rtts.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap());
            }
        }
    }
    None
}

/// Fetch measurement results from the RIPE Atlas API and convert to a DataFrame
/// matching the expected input format (addr, hostname, lat, lon, rtt).
fn fetch_atlas_measurement(measurement_id: u64, threshold: u32) -> Result<DataFrame> {
    let client = reqwest::blocking::Client::new();

    // Fetch latest measurement results
    let results_url = format!(
        "https://atlas.ripe.net/api/v2/measurements/{}/latest/?format=json",
        measurement_id
    );
    println!("Fetching latest results for RIPE Atlas measurement {}...", measurement_id);

    let response = client.get(&results_url).send()?;
    if !response.status().is_success() {
        bail!(
            "Failed to fetch measurement results: HTTP {}",
            response.status()
        );
    }
    let atlas_results: Vec<AtlasResult> = response.json()?;

    if atlas_results.is_empty() {
        bail!("No results found for measurement {}", measurement_id);
    }

    println!("Fetched {} measurement results.", atlas_results.len());

    // Collect unique probe IDs
    let probe_ids: Vec<u32> = atlas_results
        .iter()
        .map(|r| r.prb_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    println!("Fetching location data for {} probes...", probe_ids.len());

    // Fetch probe locations in batches (API supports id__in)
    let mut probe_locations: HashMap<u32, (f64, f64)> = HashMap::new();
    for chunk in probe_ids.chunks(500) {
        let ids_str: Vec<String> = chunk.iter().map(|id| id.to_string()).collect();
        let mut url = format!(
            "https://atlas.ripe.net/api/v2/probes/?id__in={}&format=json&page_size=500",
            ids_str.join(",")
        );

        loop {
            let resp = client.get(&url).send()?;
            if !resp.status().is_success() {
                bail!("Failed to fetch probe data: HTTP {}", resp.status());
            }
            let probe_resp: ProbeResponse = resp.json()?;

            for probe in &probe_resp.results {
                if let Some(ref geom) = probe.geometry {
                    if let Some(ref coords) = geom.coordinates {
                        if coords.len() == 2 {
                            probe_locations.insert(probe.id, (coords[1], coords[0])); // lat, lon
                        }
                    }
                }
            }

            match probe_resp.next {
                Some(next_url) if !next_url.is_empty() => url = next_url,
                _ => break,
            }
        }
    }

    println!(
        "Got locations for {}/{} probes.",
        probe_locations.len(),
        probe_ids.len()
    );

    // Build columns: addr, hostname, lat, lon, rtt
    let mut addrs: Vec<String> = Vec::new();
    let mut hostnames: Vec<String> = Vec::new();
    let mut lats: Vec<f32> = Vec::new();
    let mut lons: Vec<f32> = Vec::new();
    let mut rtts: Vec<f32> = Vec::new();

    for result in &atlas_results {
        let dst = match &result.dst_addr {
            Some(d) => d.clone(),
            None => continue,
        };

        // Get RTT: for ping use min, for traceroute extract from hops
        let rtt = match &result.measurement_type {
            Some(t) if t == "traceroute" => {
                result
                    .result
                    .as_ref()
                    .and_then(|r| extract_traceroute_min_rtt(r))
            }
            _ => result.min,
        };

        let rtt = match rtt {
            Some(r) if r > 0.0 => r,
            _ => continue, // skip failed measurements
        };

        // Apply RTT threshold
        if threshold > 0 && rtt > threshold as f64 {
            continue;
        }

        let (lat, lon) = match probe_locations.get(&result.prb_id) {
            Some(loc) => *loc,
            None => continue, // skip probes without location
        };

        addrs.push(dst);
        hostnames.push(format!("probe-{}", result.prb_id));
        lats.push(lat as f32);
        lons.push(lon as f32);
        rtts.push(rtt as f32);
    }

    if addrs.is_empty() {
        bail!("No valid measurement results after filtering.");
    }

    // Build DataFrame matching the expected schema
    let df = DataFrame::new(addrs.len(), vec![
        Series::new("addr".into(), addrs).into(),
        Series::new("hostname".into(), hostnames).into(),
        Series::new("lat".into(), lats).into(),
        Series::new("lon".into(), lons).into(),
        Series::new("rtt".into(), rtts).into(),
    ])?;

    // Add calculated fields (same as load_input_data)
    let df = df
        .lazy()
        .with_columns([
            col("lat").radians().alias("lat_rad"),
            col("lon").radians().alias("lon_rad"),
            (col("rtt") * lit(0.001) * lit(SPEED_OF_LIGHT) / lit(FIBER_RI) / lit(2.0))
                .alias("radius"),
        ])
        .collect()?;

    println!(
        "Loaded {} latency measurements after applying RTT threshold filter.",
        df.height()
    );

    Ok(df)
}

/// Load and preprocess airports data from a reader.
/// The airports data is expected to be in a tab-separated format with a header.
/// Expected columns: iata, size, name, lat, lon, country_code, city, pop
/// Additional fields lat_rad and lon_rad are computed in radians.
/// Parameters:
/// * reader: Any type implementing Read (file or in-memory cursor)
/// Returns:
/// * Result<Vec<Airport>>: List of airports, or error if loading fails
fn load_airports<R: polars::io::mmap::MmapBytesReader>(reader: R) -> Result<Vec<Airport>> {
    // Define the schema for the airports data
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

    // Specify CSV read options
    let airports_read_options = CsvReadOptions {
        has_header: true,
        schema: Some(airports_schema),
        parse_options: Arc::new(CsvParseOptions::default().with_separator(b'\t')),
        ..Default::default()
    };

    let airports_df = CsvReader::new(reader)
        .with_options(airports_read_options)
        .finish()?
        .lazy();

    // Compute radians for lat and lon
    let airports_df = airports_df
        .with_columns([
            col("lat").radians().alias("lat_rad"),
            col("lon").radians().alias("lon_rad"),
        ])
        .collect()?;

    // Extract columns into typed Series for building the structs.
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

/// Load and preprocess input data from the given path.
/// The input data is expected to be in a CSV format with a header.
/// Expected columns: target, hostname, lat, lon, rtt
/// Additional fields lat_rad, lon_rad, and radius are computed.
/// Parameters:
/// * path: Path to the input CSV file (PathBuf)
/// * threshold: RTT threshold to filter discs (u32)
/// Returns:
/// * Result<DataFrame>: DataFrame containing the preprocessed input data, or error if loading fails
fn load_input_data(path: &PathBuf, threshold: u32) -> Result<DataFrame> {
    // Define input schema and read options
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

    // Apply RTT threshold if provided
    if threshold > 0 {
        in_df = in_df.filter(col("rtt").lt_eq(lit(threshold as f32)));
    }

    // Add calculated fields
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

    // Validate mutually exclusive input modes
    if args.input.is_some() && args.atlas.is_some() {
        bail!("--input and --atlas are mutually exclusive. Use one or the other.");
    }
    if args.input.is_none() && args.atlas.is_none() {
        bail!("Either --input or --atlas must be provided.");
    }

    // Parse atlas measurement ID if provided (do this early for default output path)
    let atlas_id = match &args.atlas {
        Some(atlas_input) => Some(parse_atlas_id(atlas_input)?),
        None => None,
    };

    // Determine output path
    let output_path = match args.output {
        Some(p) => p,
        None => match atlas_id {
            Some(id) => PathBuf::from(format!("atlas_{}.csv", id)),
            None => bail!("--output is required when using --input."),
        },
    };

    let airports = match args.dataset.as_str() {
        "airports" => {
            println!("Using embedded airports dataset.");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_AIRPORTS)?))?
        }
        // TODO embed only the cities500 file, allow for any population threshold
        // TODO threshold should support relative numbers (e.g., Guam has no high-population city and would often result in NoCity with an absolute threshold)
        "cities500" => {
            println!("Using embedded cities500 dataset (cities with population >= 500).");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES500)?))?
        }
        "cities1000" => {
            println!("Using embedded cities1000 dataset (cities with population >= 1,000).");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES1_000)?))?
        }
        "cities5000" => {
            println!("Using embedded cities5000 dataset (cities with population >= 5,000).");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES5_000)?))?
        }
        "cities15000" => {
            println!("Using embedded cities15000 dataset (cities with population >= 15,000).");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES15_000)?))?
        }
        "cities100000" => {
            println!("Using embedded cities15000 dataset (cities with population >= 100,000).");
            load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES100_000)?))?
        }
        custom_path => {
            println!("Loading custom dataset from: {}", custom_path);
            load_airports(File::open(custom_path)?)?
        }
    };
    println!("Loaded {} locations.", airports.len());

    // Load input data from either CSV file or RIPE Atlas API
    let in_df = if let Some(ref input_path) = args.input {
        println!("Loading input data from: {:?}", input_path);
        load_input_data(input_path, args.threshold)?
    } else {
        fetch_atlas_measurement(atlas_id.unwrap(), args.threshold)?
    };

    // Group input data by target IP (each group represents one target)
    let groups_df = in_df.group_by(["addr"])?.groups()?;
    let group_indices = groups_df.column("groups")?.list()?;

    let num_targets = group_indices.len();
    println!(
        "Starting parallel processing for {} targets...",
        num_targets
    );

    // Setup progress bar
    let pb = ProgressBar::new(num_targets as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    // Process each group in parallel using Rayon
    let results: Vec<OutputRecord> = group_indices
        .into_iter() // 1. Get a standard, sequential iterator over the groups
        .par_bridge() // 2. THIS IS THE FIX: Bridge the sequential iterator to a parallel one
        .progress_with(pb)
        .filter_map(|opt_indices_series| {
            // The rest of your logic remains EXACTLY THE SAME
            opt_indices_series.map(|indices_series| {
                let indices_ca = indices_series.u32().unwrap();
                let group_df = in_df.take(indices_ca).unwrap();

                // Extract columns as Series
                let target = group_df.column("addr").unwrap().str().unwrap();
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
                    })
                    .collect();

                // Create analyzer instance and run iGreedy algorithm
                let analyzer = AnycastAnalyzer::new(discs, &airports, args.alpha, args.anycast);
                analyzer.analyze()
            })
        })
        .flatten()
        .collect();

    println!(
        "Analysis complete. Found {} geolocated sites (unicast + anycast).",
        results.len()
    );

    if !results.is_empty() {
        println!("Saving results to {:?}...", output_path);
        let num_results = results.len();
        let mut output_df = DataFrame::new(num_results, vec![
            Series::new(
                "addr".into(),
                results
                    .iter()
                    .map(|r| r.target.as_str())
                    .collect::<Vec<_>>(),
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Load the embedded airports dataset (same one the binary uses).
    fn test_airports() -> Vec<Airport> {
        load_airports(Cursor::new(decompress_gz(EMBEDDED_AIRPORTS).expect("decompress airports"))).expect("embedded airports must parse")
    }

    /// Build a `Disc` from a vantage-point location (degrees) and an RTT (ms).
    /// The radius is derived with the same formula the production code uses.
    fn make_disc(target: &str, hostname: &str, lat: f32, lon: f32, rtt: f32) -> Disc {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let radius = rtt * 0.001 * SPEED_OF_LIGHT / FIBER_RI / 2.0;
        Disc {
            target: target.to_string(),
            hostname: hostname.to_string(),
            vp_lat: lat,
            vp_lon: lon,
            rtt,
            lat_rad,
            lon_rad,
            radius,
        }
    }

    /// Collect the set of geolocated IATA codes from the output records.
    fn iata_set(results: &[OutputRecord]) -> std::collections::HashSet<String> {
        results.iter().map(|r| r.pop_iata.clone()).collect()
    }

    /// The Johannesburg area has three airports (JNB, QRA, HLA) that all
    /// share the same population value, making the scoring tiebreaker
    /// non-deterministic among them. Any of the three is a correct result.
    const JOHANNESBURG_IATAS: &[&str] = &["JNB", "QRA", "HLA"];

    fn has_johannesburg(results: &[OutputRecord]) -> bool {
        results.iter().any(|r| JOHANNESBURG_IATAS.contains(&r.pop_iata.as_str()))
    }

    // ── Airport coordinates (degrees) from the embedded dataset ─────────
    //
    // AMX  Amsterdam     52.3086,   4.7639
    // BFI  Seattle       47.5300, -122.3020
    // JNB  Johannesburg -26.1392,  28.2460  (also QRA, HLA — same city/pop)
    //
    // Nearby VPs used in case 3:
    //   LHR  London        51.4706,  -0.4619   (~355 km from AMX)
    //   BRU  Brussels      50.9014,   4.4844   (~186 km from AMX)
    //   PDX  Portland      45.5887, -122.5980  (~215 km from BFI)
    //   YVR  Vancouver     49.1939, -123.1840  (~196 km from BFI)
    //   DUR  Durban       -29.6144,  31.1197   (~398 km from JNB)
    //   CPT  Cape Town    -33.9648,  18.6017   (~1263 km from JNB)

    // ── Case 1: unicast ─────────────────────────────────────────────────
    // Single 1ms measurement centred on Amsterdam.
    // Enumeration = 1 (unicast) → should geolocate to AMX.

    #[test]
    fn case1_unicast_enumeration() {
        let airports = test_airports();
        let discs = vec![
            make_disc("1.2.3.4", "vp-ams", 52.3086, 4.7639, 1.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let (num_sites, _) = analyzer.enumeration();
        assert_eq!(num_sites, 1, "single disc ⇒ MIS size must be 1");
    }

    #[test]
    fn case1_unicast_geolocation() {
        let airports = test_airports();
        let discs = vec![
            make_disc("1.2.3.4", "vp-ams", 52.3086, 4.7639, 1.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let results = analyzer.analyze();

        assert_eq!(results.len(), 1, "unicast must produce exactly 1 result");
        assert_eq!(results[0].pop_iata, "AMX", "must geolocate to Amsterdam (AMX)");
    }

    #[test]
    fn case1_unicast_skipped_with_anycast_flag() {
        let airports = test_airports();
        let discs = vec![
            make_disc("1.2.3.4", "vp-ams", 52.3086, 4.7639, 1.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, true);
        let results = analyzer.analyze();

        assert!(results.is_empty(), "--anycast flag must skip unicast targets");
    }

    // ── Case 2: three-site anycast (one VP per site, 1ms each) ──────────
    // Three 1ms discs centred right on AMX, BFI, JNB.
    // ~98 km radius each, thousands of km apart → MIS = 3.

    #[test]
    fn case2_three_sites_enumeration() {
        let airports = test_airports();
        let discs = vec![
            make_disc("9.9.9.9", "vp-ams", 52.3086, 4.7639, 1.0),
            make_disc("9.9.9.9", "vp-sea", 47.5300, -122.3020, 1.0),
            make_disc("9.9.9.9", "vp-jnb", -26.1392, 28.2460, 1.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let (num_sites, _) = analyzer.enumeration();
        assert_eq!(num_sites, 3, "three far-apart 1ms discs ⇒ MIS size must be 3");
    }

    #[test]
    fn case2_three_sites_geolocation() {
        let airports = test_airports();
        let discs = vec![
            make_disc("9.9.9.9", "vp-ams", 52.3086, 4.7639, 1.0),
            make_disc("9.9.9.9", "vp-sea", 47.5300, -122.3020, 1.0),
            make_disc("9.9.9.9", "vp-jnb", -26.1392, 28.2460, 1.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let results = analyzer.analyze();

        let iatas = iata_set(&results);
        assert_eq!(results.len(), 3, "must produce exactly 3 results");
        assert!(iatas.contains("AMX"), "must contain Amsterdam (AMX)");
        assert!(iatas.contains("BFI"), "must contain Seattle (BFI)");
        assert!(has_johannesburg(&results), "must contain a Johannesburg airport");
    }

    // ── Case 3: three-site anycast with multiple overlapping VPs ────────
    // Each site has 3 VPs whose discs overlap locally but NOT across sites.
    //
    // Amsterdam cluster:
    //   VP at AMX   (1 ms, r ≈  99 km)
    //   VP at BRU   (3 ms, r ≈ 296 km) — 186 km from AMX, overlaps
    //   VP at LHR   (5 ms, r ≈ 493 km) — 355 km from AMX, overlaps
    //
    // Seattle cluster:
    //   VP at BFI   (1 ms, r ≈  99 km)
    //   VP at PDX   (4 ms, r ≈ 394 km) — 215 km from BFI, overlaps
    //   VP at YVR   (4 ms, r ≈ 394 km) — 196 km from BFI, overlaps
    //
    // Johannesburg cluster:
    //   VP at JNB   (1 ms, r ≈  99 km)
    //   VP at DUR   (6 ms, r ≈ 592 km) — 398 km from JNB, overlaps
    //   VP at CPT  (15 ms, r ≈1479 km) — 1263 km from JNB, overlaps

    #[test]
    fn case3_multi_vp_enumeration() {
        let airports = test_airports();
        let discs = vec![
            // Amsterdam cluster
            make_disc("8.8.8.8", "vp-ams", 52.3086, 4.7639, 1.0),
            make_disc("8.8.8.8", "vp-bru", 50.9014, 4.4844, 3.0),
            make_disc("8.8.8.8", "vp-lhr", 51.4706, -0.4619, 5.0),
            // Seattle cluster
            make_disc("8.8.8.8", "vp-sea", 47.5300, -122.3020, 1.0),
            make_disc("8.8.8.8", "vp-pdx", 45.5887, -122.5980, 4.0),
            make_disc("8.8.8.8", "vp-yvr", 49.1939, -123.1840, 4.0),
            // Johannesburg cluster
            make_disc("8.8.8.8", "vp-jnb", -26.1392, 28.2460, 1.0),
            make_disc("8.8.8.8", "vp-dur", -29.6144, 31.1197, 6.0),
            make_disc("8.8.8.8", "vp-cpt", -33.9648, 18.6017, 15.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let (num_sites, _) = analyzer.enumeration();
        assert_eq!(num_sites, 3, "three clusters of overlapping discs ⇒ MIS size must be 3");
    }

    #[test]
    fn case3_multi_vp_geolocation() {
        let airports = test_airports();
        let discs = vec![
            // Amsterdam cluster
            make_disc("8.8.8.8", "vp-ams", 52.3086, 4.7639, 1.0),
            make_disc("8.8.8.8", "vp-bru", 50.9014, 4.4844, 3.0),
            make_disc("8.8.8.8", "vp-lhr", 51.4706, -0.4619, 5.0),
            // Seattle cluster
            make_disc("8.8.8.8", "vp-sea", 47.5300, -122.3020, 1.0),
            make_disc("8.8.8.8", "vp-pdx", 45.5887, -122.5980, 4.0),
            make_disc("8.8.8.8", "vp-yvr", 49.1939, -123.1840, 4.0),
            // Johannesburg cluster
            make_disc("8.8.8.8", "vp-jnb", -26.1392, 28.2460, 1.0),
            make_disc("8.8.8.8", "vp-dur", -29.6144, 31.1197, 6.0),
            make_disc("8.8.8.8", "vp-cpt", -33.9648, 18.6017, 15.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let results = analyzer.analyze();

        let iatas = iata_set(&results);
        assert_eq!(results.len(), 3, "must produce exactly 3 results");
        assert!(iatas.contains("AMX"), "must contain Amsterdam (AMX)");
        assert!(iatas.contains("BFI"), "must contain Seattle (BFI)");
        assert!(has_johannesburg(&results), "must contain a Johannesburg airport");
    }

    // ── Case 3 addendum: intersection tightens geolocation ──────────────
    // Verify that cluster intersection doesn't accidentally widen results
    // (same 3 sites, same 3 IATAs — no extra phantom sites).

    #[test]
    fn case3_no_phantom_sites() {
        let airports = test_airports();
        let discs = vec![
            make_disc("8.8.8.8", "vp-ams", 52.3086, 4.7639, 1.0),
            make_disc("8.8.8.8", "vp-bru", 50.9014, 4.4844, 3.0),
            make_disc("8.8.8.8", "vp-lhr", 51.4706, -0.4619, 5.0),
            make_disc("8.8.8.8", "vp-sea", 47.5300, -122.3020, 1.0),
            make_disc("8.8.8.8", "vp-pdx", 45.5887, -122.5980, 4.0),
            make_disc("8.8.8.8", "vp-yvr", 49.1939, -123.1840, 4.0),
            make_disc("8.8.8.8", "vp-jnb", -26.1392, 28.2460, 1.0),
            make_disc("8.8.8.8", "vp-dur", -29.6144, 31.1197, 6.0),
            make_disc("8.8.8.8", "vp-cpt", -33.9648, 18.6017, 15.0),
        ];
        let analyzer = AnycastAnalyzer::new(discs, &airports, 1.0, false);
        let results = analyzer.analyze();

        // Every result must be one of the expected airports
        let valid: &[&str] = &["AMX", "BFI", "JNB", "QRA", "HLA"];
        for r in &results {
            assert!(
                valid.contains(&r.pop_iata.as_str()),
                "unexpected airport {} in results", r.pop_iata
            );
        }
    }
}
