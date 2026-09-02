//! RIPE Atlas probe selection.
//!
//! Determines which probes to use with `--measure`.
//!
//! * **Coverage** — probes are selected to be as widely spread as possible, since a
//!   site is only found by a probe that reaches it ([`select_spread`]).
//! * **CDN probes** — probes on well-connected networks measure closer to the speed-of-light bound,
//!   so selection prefers them ([`WELL_CONNECTED_ASNS`]).

use anyhow::{Context, Result, bail};

use crate::geo::haversine_batch;

/// Ensures well-connected probes win, unless a 'plain' one sits meaningfully farther out.
const WELL_CONNECTED_BONUS: f32 = 1.25;

/// Autonomous systems whose probes sit in a data centre rather than behind a
/// consumer access line.
const WELL_CONNECTED_ASNS: &[u32] = &[
    7224,   // Amazon
    8075,   // Microsoft Azure
    8987,   // Amazon (AWS Europe)
    12876,  // Scaleway
    13335,  // Cloudflare
    14061,  // DigitalOcean
    14618,  // Amazon (AWS)
    15169,  // Google
    16276,  // OVH
    16509,  // Amazon (AWS)
    16625,  // Akamai
    20473,  // Vultr
    20940,  // Akamai
    24940,  // Hetzner
    31898,  // Oracle Cloud
    36351,  // IBM Cloud
    45102,  // Alibaba Cloud
    51167,  // Contabo
    54113,  // Fastly
    60781,  // LeaseWeb
    63949,  // Akamai (Linode)
    197540, // netcup
    396982, // Google Cloud
];

/// A RIPE Atlas probe with a known, self-reported location.
#[derive(Debug, Clone)]
pub struct AtlasProbe {
    pub id: u32,
    pub lat: f32,
    pub lon: f32,
    pub lat_rad: f32,
    pub lon_rad: f32,
    /// Country the probe says it is in (ISO 3166-1 alpha-2), when reported.
    pub country_code: Option<String>,
    /// Anchors are hosted and maintained by RIPE NCC rather than by a volunteer.
    pub is_anchor: bool,
    /// The autonomous system the probe measures from, for the address family in use.
    pub asn: Option<u32>,
}

impl AtlasProbe {
    pub fn new(
        id: u32,
        lat: f32,
        lon: f32,
        country_code: Option<String>,
        is_anchor: bool,
        asn: Option<u32>,
    ) -> Self {
        Self {
            id,
            lat,
            lon,
            lat_rad: lat.to_radians(),
            lon_rad: lon.to_radians(),
            country_code,
            is_anchor,
            asn,
        }
    }

    /// Whether this probe is data-center hosted or an anchor.
    pub fn is_well_connected(&self) -> bool {
        self.is_anchor
            || self
                .asn
                .is_some_and(|asn| WELL_CONNECTED_ASNS.binary_search(&asn).is_ok())
    }
}

/// Read an explicit probe set from `--probes`.
///
/// Accepts either the IDs themselves (`1,2,3`) or the path of a file listing them,
/// one per line or comma-separated.
pub fn parse_probe_list(value: &str) -> Result<Vec<u32>> {
    if let Some(ids) = parse_ids(value) {
        return finish_probe_list(ids, value);
    }

    let contents = std::fs::read_to_string(value).with_context(|| {
        format!("--probes value {value:?} is neither a list of probe IDs nor a readable file")
    })?;
    let ids = parse_ids(&contents)
        .with_context(|| format!("file {value:?} contains something other than probe IDs"))?;
    finish_probe_list(ids, value)
}

/// Parse a comma- or whitespace-separated list of IDs, or `None` if anything is not one.
fn parse_ids(text: &str) -> Option<Vec<u32>> {
    let mut ids = Vec::new();
    for token in text.split([',', '\n', '\r', '\t', ' ']) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        ids.push(token.parse::<u32>().ok()?);
    }
    (!ids.is_empty()).then_some(ids)
}

/// Drop duplicate IDs while keeping the order they were given in.
fn finish_probe_list(ids: Vec<u32>, source: &str) -> Result<Vec<u32>> {
    let mut seen = std::collections::HashSet::new();
    let unique: Vec<u32> = ids.into_iter().filter(|id| seen.insert(*id)).collect();
    if unique.is_empty() {
        bail!("No probe IDs found in {source:?}.");
    }
    Ok(unique)
}

/// Pick `n` points spread as widely as possible over the globe.
///
/// This is greedy farthest-point sampling.
/// `weights`, allows for favoring well-connected probes.
pub fn farthest_point_indices(
    lats: &[f32],
    lons: &[f32],
    weights: Option<&[f32]>,
    n: usize,
) -> Vec<usize> {
    debug_assert_eq!(lats.len(), lons.len());
    debug_assert!(weights.is_none_or(|w| w.len() == lats.len()));
    let total = lats.len();
    let n = n.min(total);
    if n == 0 {
        return Vec::new();
    }

    let mut dists = vec![0.0f32; total];

    // Seed with the point farthest from the first one
    haversine_batch(lats[0], lons[0], lats, lons, &mut dists);
    let mut current = argmax(&dists, weights);

    let mut selected = Vec::with_capacity(n);
    let mut min_dist = vec![f32::INFINITY; total];

    loop {
        selected.push(current);
        if selected.len() == n {
            return selected;
        }

        // Distance from every point to the newly selected one...
        haversine_batch(lats[current], lons[current], lats, lons, &mut dists);
        // ...folded into the distance to the nearest selected point so far.
        for i in 0..total {
            if dists[i] < min_dist[i] {
                min_dist[i] = dists[i];
            }
        }
        min_dist[current] = f32::NEG_INFINITY; // never pick the same point twice

        current = argmax(&min_dist, weights);
    }
}

/// Index of the largest (optionally weighted) value
fn argmax(values: &[f32], weights: Option<&[f32]>) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &value) in values.iter().enumerate() {
        // Weighting an already-eliminated point must not resurrect it.
        let score = match weights {
            Some(w) if value.is_finite() => value * w[i],
            _ => value,
        };
        if score > best_v {
            best_i = i;
            best_v = score;
        }
    }
    best_i
}

/// Select `n` probes with the widest possible geographic spread, favouring probes
/// on well-connected networks where that costs little coverage.
pub fn select_spread(mut probes: Vec<AtlasProbe>, n: usize) -> Vec<AtlasProbe> {
    probes.sort_unstable_by_key(|p| p.id);
    if probes.len() <= n {
        return probes;
    }

    let lats: Vec<f32> = probes.iter().map(|p| p.lat_rad).collect();
    let lons: Vec<f32> = probes.iter().map(|p| p.lon_rad).collect();
    let weights: Vec<f32> = probes
        .iter()
        .map(|p| {
            if p.is_well_connected() {
                WELL_CONNECTED_BONUS
            } else {
                1.0
            }
        })
        .collect();

    let mut chosen: Vec<AtlasProbe> = farthest_point_indices(&lats, &lons, Some(&weights), n)
        .into_iter()
        .map(|i| probes[i].clone())
        .collect();
    chosen.sort_unstable_by_key(|p| p.id);
    chosen
}
