use anyhow::{bail, Result};
use polars::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;

use crate::geo::{FIBER_RI, SPEED_OF_LIGHT};

/// Deserialize a value that may be a number or a numeric string into Option<f64>.
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

#[derive(Deserialize)]
struct ProbeGeometry {
    coordinates: Option<Vec<f64>>,
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
pub fn parse_atlas_id(input: &str) -> Result<u64> {
    if let Ok(id) = input.trim().parse::<u64>() {
        return Ok(id);
    }

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
pub fn fetch_atlas_measurement(measurement_id: u64, threshold: u32) -> Result<DataFrame> {
    let client = reqwest::blocking::Client::new();

    let results_url = format!(
        "https://atlas.ripe.net/api/v2/measurements/{}/latest/?format=json",
        measurement_id
    );
    println!(
        "Fetching latest results for RIPE Atlas measurement {}...",
        measurement_id
    );

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

    let probe_ids: Vec<u32> = atlas_results
        .iter()
        .map(|r| r.prb_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    println!("Fetching location data for {} probes...", probe_ids.len());

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
                            probe_locations.insert(probe.id, (coords[1], coords[0]));
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

        let rtt = match &result.measurement_type {
            Some(t) if t == "traceroute" => result
                .result
                .as_ref()
                .and_then(|r| extract_traceroute_min_rtt(r)),
            _ => result.min,
        };

        let rtt = match rtt {
            Some(r) if r > 0.0 => r,
            _ => continue,
        };

        if threshold > 0 && rtt > threshold as f64 {
            continue;
        }

        let (lat, lon) = match probe_locations.get(&result.prb_id) {
            Some(loc) => *loc,
            None => continue,
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

    let df = DataFrame::new(addrs.len(), vec![
        Series::new("addr".into(), addrs).into(),
        Series::new("hostname".into(), hostnames).into(),
        Series::new("lat".into(), lats).into(),
        Series::new("lon".into(), lons).into(),
        Series::new("rtt".into(), rtts).into(),
    ])?;

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
