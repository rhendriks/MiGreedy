//! RIPE Atlas integration.
//!
//! Two options:
//!
//! * `--atlas <ID>` reads the results of a measurement that already exists
//!   ([`fetch_atlas_measurement`]);
//! * `--measure <TARGET>` schedules a new one-off ping from probes MiGreedy picks
//!   itself and waits for the results ([`run_measurement`]).

use anyhow::{Context, Result, bail};
use indicatif::ProgressBar;
use polars::prelude::*;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::time::{Duration, Instant};

use crate::geo::{haversine_distance, rtt_to_radius_km};
use crate::io::{finalize_measurements, progress_bar};
use crate::probes::{AtlasProbe, farthest_point_indices, select_spread};

/// Attach MiGreedy descriptions to RIPE Atlas measurements created
const DESCRIPTION_PREFIX: &str = concat!(
    "MiGreedy v",
    env!("CARGO_PKG_VERSION"),
    " anycast geolocation"
);

const API_HOST: &str = "https://atlas.ripe.net/";
const API_BASE: &str = "https://atlas.ripe.net/api/v2";

/// Largest page the RIPE Atlas API will serve.
const PAGE_SIZE: usize = 500;

/// How often a scheduled measurement is checked for completion.
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// How often the wait is reported, so a long one does not scroll the terminal.
const REPORT_INTERVAL: Duration = Duration::from_secs(30);

/// Anchors used to check probe locations against the speed of light.
const VALIDATION_ANCHORS: usize = 5;

/// Slack allowed before a probe is called out for an impossible location.
const VALIDATION_SLACK_KM: f32 = 100.0;

/// Additional probes sampled when they are verified, as some will be invalid.
const VALIDATION_OVERSAMPLE: f32 = 1.1;

/// Deserialize a value that may be a number or a numeric string into `Option<f64>`.
/// The RIPE Atlas API inconsistently returns some fields as strings.
fn deserialize_f64_or_string<'de, D>(deserializer: D) -> std::result::Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(serde_json::Value::Number(n)) => Ok(n.as_f64()),
        Some(serde_json::Value::String(s)) => Ok(s.parse::<f64>().ok()),
        _ => Ok(None),
    }
}

#[derive(Deserialize)]
struct AtlasResult {
    dst_addr: Option<String>,
    #[serde(default, deserialize_with = "deserialize_f64_or_string")]
    min: Option<f64>,
    prb_id: u32,
    #[serde(rename = "type")]
    measurement_type: Option<String>,
    result: Option<serde_json::Value>,
}

impl AtlasResult {
    /// The RTT this result contributes, whatever measurement type produced it.
    fn rtt(&self) -> Option<f64> {
        match self.measurement_type.as_deref() {
            Some("traceroute") => self.result.as_ref().and_then(extract_traceroute_min_rtt),
            Some("dns") => self.result.as_ref().and_then(extract_dns_rtt),
            _ => self.min,
        }
    }
}

#[derive(Deserialize)]
struct ProbeGeometry {
    coordinates: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct ProbeInfo {
    id: u32,
    geometry: Option<ProbeGeometry>,
    #[serde(default)]
    country_code: Option<String>,
    #[serde(default)]
    is_anchor: bool,
    #[serde(default)]
    asn_v4: Option<u32>,
    #[serde(default)]
    asn_v6: Option<u32>,
    #[serde(default)]
    address_v4: Option<String>,
    #[serde(default)]
    address_v6: Option<String>,
}

impl ProbeInfo {
    /// `(latitude, longitude)` in degrees, if the probe reports a location.
    fn coordinates(&self) -> Option<(f64, f64)> {
        let coords = self.geometry.as_ref()?.coordinates.as_ref()?;
        // GeoJSON orders coordinates longitude-first.
        (coords.len() == 2).then(|| (coords[1], coords[0]))
    }

    /// The probe as MiGreedy models it, carrying the AS it measures `af` from.
    fn into_probe(self, af: u8) -> Option<AtlasProbe> {
        let (lat, lon) = self.coordinates()?;
        let asn = match af {
            6 => self.asn_v6,
            _ => self.asn_v4,
        };
        Some(AtlasProbe::new(
            self.id,
            lat as f32,
            lon as f32,
            self.country_code.clone(),
            self.is_anchor,
            asn,
        ))
    }
}

#[derive(Deserialize)]
struct ProbeResponse {
    results: Vec<ProbeInfo>,
    next: Option<String>,
    /// Total across all pages, reported on every page.
    #[serde(default)]
    count: Option<u64>,
}

#[derive(Deserialize)]
struct MeasurementStatus {
    id: u32,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct MeasurementInfo {
    status: MeasurementStatus,
}

#[derive(Deserialize)]
struct CreateResponse {
    measurements: Vec<u64>,
}

/// Extract the measurement ID from either a plain numeric string or a RIPE Atlas URL.
pub fn parse_atlas_id(input: &str) -> Result<u64> {
    if let Ok(id) = input.trim().parse::<u64>() {
        return Ok(id);
    }

    let parts: Vec<&str> = input.trim_end_matches('/').split('/').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "measurements"
            && let Some(id_str) = parts.get(i + 1)
            && let Ok(id) = id_str.parse::<u64>()
        {
            return Ok(id);
        }
    }

    bail!("Could not parse RIPE Atlas measurement ID from: {}", input)
}

/// Extract RTT from a DNS result (stored in result.rt).
fn extract_dns_rtt(result_value: &serde_json::Value) -> Option<f64> {
    result_value.get("rt").and_then(|v| v.as_f64())
}

/// Extract minimum RTT from a traceroute result by finding the last hop with valid RTT.
fn extract_traceroute_min_rtt(result_value: &serde_json::Value) -> Option<f64> {
    let hops = result_value.as_array()?;
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

/// Build the POST body that creates one one-off ping per target.
fn ping_request_body(
    targets: &[String],
    af: u8,
    probe_ids: &[u32],
    packets: u8,
    purpose: &str,
) -> serde_json::Value {
    let definitions: Vec<serde_json::Value> = targets
        .iter()
        .map(|target| {
            serde_json::json!({
                "target": target,
                "af": af,
                "type": "ping",
                "packets": packets,
                "description": format!("{DESCRIPTION_PREFIX}: {purpose} ({target})"),
                "resolve_on_probe": false,
            })
        })
        .collect();

    let ids: Vec<String> = probe_ids.iter().map(|id| id.to_string()).collect();
    serde_json::json!({
        "definitions": definitions,
        "probes": [{
            "type": "probes",
            "value": ids.join(","),
            "requested": probe_ids.len(),
        }],
        "is_oneoff": true,
    })
}

/// A thin RIPE Atlas API v2 client.
///
/// The API key is only needed to *create* measurements; reading public results and
/// probe metadata works without one.
pub struct AtlasClient {
    http: reqwest::blocking::Client,
    key: Option<String>,
}

impl AtlasClient {
    pub fn new(key: Option<String>) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .user_agent(concat!("migreedy/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { http, key })
    }

    fn get(&self, url: &str) -> Result<reqwest::blocking::Response> {
        let mut request = self.http.get(url);
        // Pagination follows URLs taken from response bodies
        if let Some(ref key) = self.key
            && url.starts_with(API_HOST)
        {
            request = request.header("Authorization", format!("Key {key}"));
        }
        Ok(request.send()?)
    }

    /// GET a URL and deserialize the JSON body, surfacing API errors as messages.
    fn get_json<T: DeserializeOwned>(&self, url: &str) -> Result<T> {
        let response = self.get(url)?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            bail!(
                "RIPE Atlas API request failed (HTTP {status}): {}",
                brief(&body)
            );
        }
        Ok(response.json()?)
    }

    /// Walk a paginated probe listing, collecting every page.
    fn paged_probes(&self, first_url: String) -> Result<Vec<ProbeInfo>> {
        let mut url = first_url;
        let mut all = Vec::new();
        let mut progress: Option<ProgressBar> = None;

        loop {
            let page: ProbeResponse = self.get_json(&url)?;

            // Progress bar
            if progress.is_none()
                && let Some(count) = page.count.filter(|c| *c as usize > PAGE_SIZE * 2)
            {
                progress = Some(progress_bar(count)?);
            }
            all.extend(page.results);
            if let Some(ref bar) = progress {
                bar.set_position(all.len() as u64);
            }

            match page.next {
                Some(next) if !next.is_empty() => url = next,
                _ => {
                    if let Some(bar) = progress {
                        bar.finish_and_clear();
                    }
                    return Ok(all);
                }
            }
        }
    }

    /// Locations of specific probes, keyed by probe ID.
    fn probe_locations(&self, ids: &[u32]) -> Result<HashMap<u32, (f64, f64)>> {
        let mut locations = HashMap::new();
        for chunk in ids.chunks(PAGE_SIZE) {
            let ids_str: Vec<String> = chunk.iter().map(|id| id.to_string()).collect();
            let url = format!(
                "{API_BASE}/probes/?id__in={}&format=json&page_size={PAGE_SIZE}",
                ids_str.join(",")
            );
            for probe in self.paged_probes(url)? {
                if let Some(coords) = probe.coordinates() {
                    locations.insert(probe.id, coords);
                }
            }
        }
        Ok(locations)
    }

    /// Every connected probe that can reach the given address family.
    /// Uses the `system-ipv4-works` / `system-ipv6-works` tags.
    fn probe_catalogue(&self, af: u8) -> Result<Vec<AtlasProbe>> {
        let url = format!(
            "{API_BASE}/probes/?status=1&tags={},{}&format=json&page_size={PAGE_SIZE}\
             &fields=id,geometry,country_code,is_anchor,asn_v4,asn_v6",
            capability_tag(af),
            stability_tag(af)
        );
        println!("Fetching the RIPE Atlas probe catalogue (this takes a moment)...");
        let probes: Vec<AtlasProbe> = self
            .paged_probes(url)?
            .into_iter()
            .filter_map(|info| info.into_probe(af))
            .collect();
        if probes.is_empty() {
            bail!("RIPE Atlas returned no connected, stable IPv{af} probes.");
        }
        println!(
            "{} connected probes are stable on IPv{af} and report a location.",
            probes.len()
        );
        Ok(probes)
    }

    /// Connected anchors and their addresses, used as validation landmarks.
    fn anchors(&self, af: u8) -> Result<Vec<(AtlasProbe, String)>> {
        let url = format!(
            "{API_BASE}/probes/?status=1&is_anchor=true&format=json&page_size={PAGE_SIZE}\
             &fields=id,geometry,country_code,is_anchor,asn_v4,asn_v6,address_v4,address_v6"
        );
        let anchors: Vec<(AtlasProbe, String)> = self
            .paged_probes(url)?
            .into_iter()
            .filter_map(|info| {
                let address = match af {
                    6 => info.address_v6.clone(),
                    _ => info.address_v4.clone(),
                }
                .filter(|a| !a.is_empty())?;
                Some((info.into_probe(af)?, address))
            })
            .collect();
        Ok(anchors)
    }

    /// Schedule a one-off ping from `probe_ids` to each target.
    ///
    /// `purpose` is a description that is added to the RIPE Atlas measurement
    ///
    /// Returns one measurement ID per target, in the order the targets were given.
    fn create_ping(
        &self,
        targets: &[String],
        af: u8,
        probe_ids: &[u32],
        packets: u8,
        purpose: &str,
    ) -> Result<Vec<u64>> {
        let key = self
            .key
            .as_ref()
            .context("An API key is required to create RIPE Atlas measurements.")?;

        let body = ping_request_body(targets, af, probe_ids, packets, purpose);

        let response = self
            .http
            .post(format!("{API_BASE}/measurements/"))
            .header("Authorization", format!("Key {key}"))
            .json(&body)
            .send()?;

        let status = response.status();
        let text = response.text().unwrap_or_default();
        if !status.is_success() {
            bail!(
                "Could not create the RIPE Atlas measurement (HTTP {status}): {}",
                brief(&text)
            );
        }

        let created: CreateResponse = serde_json::from_str(&text).with_context(|| {
            format!("unexpected response creating measurement: {}", brief(&text))
        })?;
        if created.measurements.is_empty() {
            bail!("RIPE Atlas accepted the request but created no measurements.");
        }
        Ok(created.measurements)
    }

    /// Current status of a measurement, as `(status id, status name)`.
    fn status(&self, id: u64) -> Result<(u32, String)> {
        let info: MeasurementInfo = self.get_json(&format!(
            "{API_BASE}/measurements/{id}/?format=json&fields=status"
        ))?;
        let name = info
            .status
            .name
            .unwrap_or_else(|| info.status.id.to_string());
        Ok((info.status.id, name))
    }

    /// All results of a measurement.
    fn results(&self, id: u64) -> Result<Vec<AtlasResult>> {
        self.get_json(&format!(
            "{API_BASE}/measurements/{id}/results/?format=json"
        ))
    }

    /// Wait for one-off measurements to stop, then collect their results.
    /// Forcefully collects results if the `timeout` is elapsed.
    fn wait_for_results(&self, ids: &[u64], timeout: Duration) -> Result<Vec<AtlasResult>> {
        let started = Instant::now();
        let mut pending: Vec<u64> = ids.to_vec();
        let mut last_report = Instant::now();

        while !pending.is_empty() {
            std::thread::sleep(POLL_INTERVAL);

            let mut still_running = Vec::new();
            for &id in &pending {
                let (status_id, status_name) = self.status(id)?;
                match status_id {
                    // Stopped / Forced to stop / Archived: the measurement is done.
                    4 | 5 | 8 => {}
                    // No suitable probes / Failed.
                    6 | 7 => bail!("RIPE Atlas measurement {id} ended as \"{status_name}\"."),
                    _ => still_running.push(id),
                }
            }
            pending = still_running;

            if !pending.is_empty() && started.elapsed() >= timeout {
                println!(
                    "Timed out after {}s with {} measurement(s) still running; using the results collected so far.",
                    timeout.as_secs(),
                    pending.len()
                );
                break;
            }
            // Polls are frequent so the run finishes promptly; the reporting is not.
            if !pending.is_empty() && last_report.elapsed() >= REPORT_INTERVAL {
                last_report = Instant::now();
                println!(
                    "Waiting for {} measurement(s) to finish ({}s elapsed)...",
                    pending.len(),
                    started.elapsed().as_secs()
                );
            }
        }

        let mut all = Vec::new();
        for &id in ids {
            all.extend(self.results(id)?);
        }
        Ok(all)
    }
}

/// Trim an API error body to something readable in a terminal.
fn brief(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() > 400 {
        let cut: String = trimmed.chars().take(400).collect();
        format!("{cut}...")
    } else {
        trimmed.to_string()
    }
}

/// The RIPE Atlas system tag asserting a probe can measure this address family.
fn capability_tag(af: u8) -> &'static str {
    match af {
        6 => "system-ipv6-works",
        _ => "system-ipv4-works",
    }
}

/// The RIPE Atlas system tag asserting a probe is stable.
fn stability_tag(af: u8) -> &'static str {
    match af {
        6 => "system-ipv6-stable-30d",
        _ => "system-ipv4-stable-30d",
    }
}

/// Address family shared by all targets.
fn address_family(targets: &[String]) -> Result<u8> {
    let mut families: HashSet<u8> = HashSet::new();
    for target in targets {
        match target.parse::<IpAddr>() {
            Ok(IpAddr::V4(_)) => {
                families.insert(4);
            }
            Ok(IpAddr::V6(_)) => {
                families.insert(6);
            }
            Err(_) => {} // a hostname; it can be resolved to either family
        }
    }

    if families.len() > 1 {
        bail!("Cannot measure IPv4 and IPv6 targets in one run: give them in separate runs.");
    }
    Ok(families.into_iter().next().unwrap_or(4))
}

/// Turn Atlas results into the `addr, hostname, lat, lon, rtt` frame the pipeline reads.
fn results_to_frame(
    results: &[AtlasResult],
    locations: &HashMap<u32, (f64, f64)>,
    threshold: u32,
) -> Result<DataFrame> {
    let mut addrs: Vec<String> = Vec::new();
    let mut hostnames: Vec<String> = Vec::new();
    let mut lats: Vec<f32> = Vec::new();
    let mut lons: Vec<f32> = Vec::new();
    let mut rtts: Vec<f32> = Vec::new();

    for result in results {
        let Some(dst) = result.dst_addr.clone() else {
            continue;
        };
        // Filtering on RTT is left to `finalize_measurements`
        let Some(rtt) = result.rtt() else { continue };
        let Some(&(lat, lon)) = locations.get(&result.prb_id) else {
            continue;
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

    let df = DataFrame::new(
        addrs.len(),
        vec![
            Series::new("addr".into(), addrs).into(),
            Series::new("hostname".into(), hostnames).into(),
            Series::new("lat".into(), lats).into(),
            Series::new("lon".into(), lons).into(),
            Series::new("rtt".into(), rtts).into(),
        ],
    )?;

    finalize_measurements(df, threshold)
}

/// Fetch the latest results of an existing measurement and convert to a DataFrame
/// matching the expected input format (addr, hostname, lat, lon, rtt).
pub fn fetch_atlas_measurement(measurement_id: u64, threshold: u32) -> Result<DataFrame> {
    let client = AtlasClient::new(None)?;

    println!("Fetching latest results for RIPE Atlas measurement {measurement_id}...");
    let atlas_results: Vec<AtlasResult> = client.get_json(&format!(
        "{API_BASE}/measurements/{measurement_id}/latest/?format=json"
    ))?;

    if atlas_results.is_empty() {
        bail!("No results found for measurement {measurement_id}");
    }
    println!("Fetched {} measurement results.", atlas_results.len());

    let locations = fetch_result_locations(&client, &atlas_results)?;
    results_to_frame(&atlas_results, &locations, threshold)
}

/// Look up the location of every probe appearing in a result set.
fn fetch_result_locations(
    client: &AtlasClient,
    results: &[AtlasResult],
) -> Result<HashMap<u32, (f64, f64)>> {
    let probe_ids: Vec<u32> = results
        .iter()
        .map(|r| r.prb_id)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    println!("Fetching location data for {} probes...", probe_ids.len());
    let locations = client.probe_locations(&probe_ids)?;
    println!(
        "Got locations for {}/{} probes.",
        locations.len(),
        probe_ids.len()
    );
    Ok(locations)
}

/// How the probes for a new measurement are chosen.
pub enum ProbeChoice {
    /// Exactly these probe IDs.
    Explicit(Vec<u32>),
    /// This many probes, picked for the widest global spread.
    Spread(usize),
}

/// Everything `--measure` needs to schedule and collect a run.
pub struct MeasureOptions {
    pub targets: Vec<String>,
    pub probes: ProbeChoice,
    pub packets: u8,
    pub timeout: Duration,
    /// Additionally ping anchors to catch locations the speed of light rules out.
    pub validate: bool,
    /// Select and report probes without scheduling anything.
    pub dry_run: bool,
}

/// Schedule a measurement, wait for it, and return `(measurement ids, measurements)`.
///
/// With `dry_run` set nothing is scheduled and `Ok(None)` is returned after
/// reporting the probe selection.
pub fn run_measurement(
    client: &AtlasClient,
    options: &MeasureOptions,
    threshold: u32,
) -> Result<Option<(Vec<u64>, DataFrame)>> {
    if options.targets.is_empty() {
        bail!("--measure needs at least one target.");
    }
    let af = address_family(&options.targets)?;

    let probes = choose_probes(client, options, af)?;
    if probes.is_empty() {
        bail!("No usable probes remain; select probes explicitly to override.");
    }

    println!(
        "Selected {} probes for {} IPv{af} target(s): {}.",
        probes.len(),
        options.targets.len(),
        options.targets.join(", ")
    );
    report_spread(&probes);

    if options.dry_run {
        println!("\nProbe IDs: {}", join_ids(&probes));
        println!(
            "\n--dry_run: nothing was scheduled. Without it this would ping {} target(s) from {} probe(s) with {} packet(s) each, spending RIPE Atlas credits.",
            options.targets.len(),
            probes.len(),
            options.packets
        );
        if options.validate {
            println!(
                "--validate_probes was skipped: it pings anchors, which a dry run must not do. The real run will validate and may keep fewer probes."
            );
        }
        return Ok(None);
    }

    let probe_ids: Vec<u32> = probes.iter().map(|p| p.id).collect();
    println!(
        "Scheduling a one-off ping ({} packet(s)) from {} probe(s); this spends RIPE Atlas credits.",
        options.packets,
        probe_ids.len()
    );

    let ids = client.create_ping(
        &options.targets,
        af,
        &probe_ids,
        options.packets,
        "target measurement",
    )?;
    for id in &ids {
        println!("Created measurement {id}: https://atlas.ripe.net/measurements/{id}/");
    }

    let results = client.wait_for_results(&ids, options.timeout)?;
    let responded: HashSet<u32> = results.iter().map(|r| r.prb_id).collect();
    println!(
        "Collected {} results from {}/{} probes.",
        results.len(),
        responded.len(),
        probe_ids.len()
    );

    // The probe locations are already known from selection, so no refetch is needed.
    let locations: HashMap<u32, (f64, f64)> = probes
        .iter()
        .map(|p| (p.id, (p.lat as f64, p.lon as f64)))
        .collect();

    let df = results_to_frame(&results, &locations, threshold)?;
    Ok(Some((ids, df)))
}

/// Resolve the probe set: an explicit list, or a spread selection over the catalogue.
fn choose_probes(
    client: &AtlasClient,
    options: &MeasureOptions,
    af: u8,
) -> Result<Vec<AtlasProbe>> {
    match &options.probes {
        ProbeChoice::Explicit(ids) => {
            println!("Looking up the {} requested probes...", ids.len());
            let mut probes = fetch_probes_by_id(client, ids, af)?;
            let missing = ids.len().saturating_sub(probes.len());
            if missing > 0 {
                println!("{missing} requested probe(s) are unknown or report no location.");
            }
            // Validation pings anchors, so a dry run must not reach it.
            if options.validate && !options.dry_run {
                probes = validate_locations(client, probes, af, options)?;
            }
            Ok(probes)
        }
        ProbeChoice::Spread(wanted) => {
            let catalogue = client.probe_catalogue(af)?;

            // Validation pings anchors, so a dry run must not reach it.
            if options.validate && !options.dry_run {
                // Validation removes probes, so we check 1.1 times the requested probes
                let oversample = ((*wanted as f32) * VALIDATION_OVERSAMPLE).ceil() as usize;
                let candidates = select_spread(catalogue, oversample);
                let survivors = validate_locations(client, candidates, af, options)?;
                Ok(select_spread(survivors, *wanted))
            } else {
                Ok(select_spread(catalogue, *wanted))
            }
        }
    }
}

/// Fetch specific probes by ID, keeping only those that report a location.
fn fetch_probes_by_id(client: &AtlasClient, ids: &[u32], af: u8) -> Result<Vec<AtlasProbe>> {
    let mut probes = Vec::new();
    for chunk in ids.chunks(PAGE_SIZE) {
        let ids_str: Vec<String> = chunk.iter().map(|id| id.to_string()).collect();
        let url = format!(
            "{API_BASE}/probes/?id__in={}&format=json&page_size={PAGE_SIZE}\
             &fields=id,geometry,country_code,is_anchor,asn_v4,asn_v6",
            ids_str.join(",")
        );
        probes.extend(
            client
                .paged_probes(url)?
                .into_iter()
                .filter_map(|info| info.into_probe(af)),
        );
    }
    probes.sort_unstable_by_key(|p| p.id);
    Ok(probes)
}

/// Drop probes whose reported location is farther from an anchor than the measured
/// RTT to that anchor allows.
fn validate_locations(
    client: &AtlasClient,
    probes: Vec<AtlasProbe>,
    af: u8,
    options: &MeasureOptions,
) -> Result<Vec<AtlasProbe>> {
    if probes.is_empty() {
        return Ok(probes);
    }

    println!("Validating probe locations against RIPE Atlas anchors...");
    let anchors = client.anchors(af)?;
    if anchors.len() < 2 {
        println!("Not enough anchors available to validate against; skipping validation.");
        return Ok(probes);
    }

    let lats: Vec<f32> = anchors.iter().map(|(a, _)| a.lat_rad).collect();
    let lons: Vec<f32> = anchors.iter().map(|(a, _)| a.lon_rad).collect();
    let chosen: Vec<&(AtlasProbe, String)> =
        farthest_point_indices(&lats, &lons, None, VALIDATION_ANCHORS.min(anchors.len()))
            .into_iter()
            .map(|i| &anchors[i])
            .collect();

    let targets: Vec<String> = chosen.iter().map(|(_, addr)| addr.clone()).collect();
    let anchor_of_address: HashMap<&str, usize> = chosen
        .iter()
        .enumerate()
        .map(|(i, (_, addr))| (addr.as_str(), i))
        .collect();

    let probe_ids: Vec<u32> = probes.iter().map(|p| p.id).collect();
    println!(
        "Pinging {} anchors from {} probe(s) to check their locations.",
        targets.len(),
        probe_ids.len()
    );

    let ids = client.create_ping(
        &targets,
        af,
        &probe_ids,
        options.packets,
        "probe location validation",
    )?;
    let results = client.wait_for_results(&ids, options.timeout)?;

    // Keep the tightest constraint each probe gave us per anchor.
    let mut best: HashMap<(u32, usize), f64> = HashMap::new();
    for result in &results {
        let (Some(addr), Some(rtt)) = (result.dst_addr.as_deref(), result.rtt()) else {
            continue;
        };
        if rtt <= 0.0 {
            continue;
        }
        let Some(&anchor) = anchor_of_address.get(addr) else {
            continue;
        };
        best.entry((result.prb_id, anchor))
            .and_modify(|existing| *existing = existing.min(rtt))
            .or_insert(rtt);
    }

    let mut kept = Vec::with_capacity(probes.len());
    let mut impossible = 0usize;
    let mut unverified = 0usize;

    for probe in probes {
        // Always keep anchors
        if probe.is_anchor {
            kept.push(probe);
            continue;
        }

        let mut checked = 0usize;
        let mut violated = false;
        for (index, (anchor, _)) in chosen.iter().enumerate() {
            let Some(&rtt) = best.get(&(probe.id, index)) else {
                continue;
            };
            checked += 1;
            let reachable_km = rtt_to_radius_km(rtt as f32);
            let claimed_km =
                haversine_distance(probe.lat_rad, probe.lon_rad, anchor.lat_rad, anchor.lon_rad);
            if claimed_km > reachable_km + VALIDATION_SLACK_KM {
                violated = true;
                break;
            }
        }

        if violated {
            impossible += 1;
        } else {
            if checked == 0 {
                unverified += 1;
            }
            kept.push(probe);
        }
    }

    println!(
        "Validation: dropped {impossible} probe(s) whose location the RTTs rule out; {unverified} probe(s) answered no anchor and were kept unverified."
    );
    Ok(kept)
}

/// Report how evenly the selected probes cover the globe.
fn report_spread(probes: &[AtlasProbe]) {
    if probes.len() < 2 {
        return;
    }
    let mut nearest = f32::INFINITY;
    for (i, a) in probes.iter().enumerate() {
        for b in &probes[i + 1..] {
            let d = haversine_distance(a.lat_rad, a.lon_rad, b.lat_rad, b.lon_rad);
            if d < nearest {
                nearest = d;
            }
        }
    }
    let countries: HashSet<&str> = probes
        .iter()
        .filter_map(|p| p.country_code.as_deref())
        .collect();
    let well_connected = probes.iter().filter(|p| p.is_well_connected()).count();
    println!(
        "Coverage: {} countries, closest pair {:.0} km apart, {}/{} probes on well-connected networks.",
        countries.len(),
        nearest,
        well_connected,
        probes.len()
    );
}

/// Probe IDs as a comma-separated list, ready to paste back into `--probes`.
fn join_ids(probes: &[AtlasProbe]) -> String {
    probes
        .iter()
        .map(|p| p.id.to_string())
        .collect::<Vec<_>>()
        .join(",")
}
