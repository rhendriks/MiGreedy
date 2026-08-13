//! Parses a file that maps vantage points (VPs) to its locations.
//! This allows for parsing CSV/warts files without `lat`/`lon` columns for better data compression.
//!
//! The format is whitespace-separated `hostname lat lon`, one VP per line, no header:
//!
//! ```text
//! hlz2-nz.ark.caida.org -37.79 175.28
//! fra-de.ark.caida.org 50.11 8.74
//! ```
//! DNS suffixes are removed so `san-us` and `san-us.ark.caida.org` resolve to each other.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A vantage point's hostname and location.
#[derive(Debug, Clone, PartialEq)]
pub struct Vp {
    pub hostname: String,
    pub lat: f32,
    pub lon: f32,
}

/// Hostname-to-coordinates lookup for vantage points.
pub struct VpTable {
    /// Keyed on the full hostname as written in the file.
    exact: HashMap<String, Vp>,
    /// Keyed on the label before the first dot.
    by_label: HashMap<String, Vp>,
    /// Lines skipped because they were malformed.
    pub malformed: usize,
    /// Lines skipped because the hostname was already seen.
    pub duplicates: usize,
}

/// The label before the first dot, e.g. `san-us` for `san-us.ark.caida.org`.
fn label_of(hostname: &str) -> &str {
    hostname.split('.').next().unwrap_or(hostname)
}

impl VpTable {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open VPs file {}", path.display()))?;
        Self::parse(BufReader::new(file))
    }

    pub fn parse<R: BufRead>(reader: R) -> Result<Self> {
        let mut exact: HashMap<String, Vp> = HashMap::new();
        let mut label_hits: HashMap<String, Option<Vp>> = HashMap::new();
        let mut malformed = 0usize;
        let mut duplicates = 0usize;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut fields = line.split_whitespace();
            let (Some(hostname), Some(lat), Some(lon)) =
                (fields.next(), fields.next(), fields.next())
            else {
                malformed += 1;
                continue;
            };

            let (Ok(lat), Ok(lon)) = (lat.parse::<f32>(), lon.parse::<f32>()) else {
                malformed += 1;
                continue;
            };

            if !lat.is_finite() || !lon.is_finite() {
                malformed += 1;
                continue;
            }

            // The file may list a hostname more than once; keep the first occurrence.
            if exact.contains_key(hostname) {
                duplicates += 1;
                continue;
            }
            let vp = Vp {
                hostname: hostname.to_string(),
                lat,
                lon,
            };
            exact.insert(hostname.to_string(), vp.clone());

            // Record the short label if it is unique and unambiguous
            match label_hits.entry(label_of(hostname).to_string()) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(Some(vp));
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if matches!(e.get(), Some(seen) if *seen != vp) {
                        e.insert(None);
                    }
                }
            }
        }

        let by_label = label_hits
            .into_iter()
            .filter_map(|(label, vp)| vp.map(|v| (label, v)))
            .collect();

        Ok(VpTable {
            exact,
            by_label,
            malformed,
            duplicates,
        })
    }

    /// Number of vantage points with known coordinates.
    pub fn len(&self) -> usize {
        self.exact.len()
    }

    /// Look up a VP, matching on the full hostname first and its leading label second.
    pub fn get(&self, hostname: &str) -> Option<&Vp> {
        if let Some(vp) = self.exact.get(hostname) {
            return Some(vp);
        }
        self.by_label.get(label_of(hostname))
    }
}
