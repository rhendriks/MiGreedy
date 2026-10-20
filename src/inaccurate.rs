//! Filters RIPE Atlas probes with an inaccurate reported location.
//! Uses the list <https://github.com/kizhikevich/violating_ripe_probes>
//! by [Izhikevich et al.](https://arxiv.org/abs/2409.19109).
//!
//! [`load`] fetches this list for a given date.
//!
//! TODO cache these lists in the binary?

use anyhow::{Result, bail};
use chrono::{Datelike, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;

use crate::config;

const LIST_BASE: &str =
    "https://raw.githubusercontent.com/kizhikevich/violating_ripe_probes/main/probe_ids_";

/// The oldest published list. Earlier dates use this one.
const OLDEST_RELEASE: (i32, u32, u32) = (2025, 12, 1);

/// A list of probe IDs and the release it came from.
struct List {
    release: String,
    ids: HashSet<u32>,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    release: String,
    ids: Vec<u32>,
}

/// Load the list corresponding to the provided date.
/// A date before the first release gets the oldest list. `None` means today.
///
/// When no list can be fetched, a warning will be printed.
pub fn load(date: Option<NaiveDate>) -> HashSet<u32> {
    let date = date.unwrap_or_else(|| Utc::now().date_naive());
    let oldest = oldest_release();
    if date < oldest {
        println!(
            "{date} is before the first list of inaccurate probes ({oldest}); using the oldest."
        );
    }

    match find_list(date) {
        Ok(list) => {
            println!(
                "Excluding {} probes with inaccurate locations (list of {}).",
                list.ids.len(),
                list.release
            );
            list.ids
        }
        Err(e) => {
            println!(
                "Warning: could not get the inaccurate probe list ({e:#}); no probes are excluded. \
                 Pass --include_inaccurate_probes to silence this."
            );
            HashSet::new()
        }
    }
}

fn oldest_release() -> NaiveDate {
    let (y, m, d) = OLDEST_RELEASE;
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

/// Releases that can apply to `date`, newest first, down to the oldest one.
/// Releases are on the 1st and 15th.
fn releases_up_to(date: NaiveDate) -> Vec<NaiveDate> {
    let oldest = oldest_release();
    if date < oldest {
        return vec![oldest];
    }

    let mut dates = Vec::new();
    let mut next = if date.day() >= 15 {
        date.with_day(15)
    } else {
        date.with_day(1)
    };
    while let Some(d) = next
        && d >= oldest
    {
        dates.push(d);
        // From the 15th the previous release is the 1st; from the 1st it is the 15th of last month.
        next = if d.day() == 15 {
            d.with_day(1)
        } else {
            d.pred_opt().and_then(|last| last.with_day(15))
        };
    }
    dates
}

/// Fetches the list for the newest release on or before `date`.
/// Lists are cached.
fn find_list(date: NaiveDate) -> Result<List> {
    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(concat!("migreedy/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let mut network_error: Option<anyhow::Error> = None;

    for release_date in releases_up_to(date) {
        let release = release_date.format("%Y%m%d").to_string();
        // Check the cache first
        if let Some(list) = read_cache(&release) {
            return Ok(list);
        }
        if network_error.is_some() {
            continue;
        }

        let response = match http.get(format!("{LIST_BASE}{release}.txt")).send() {
            Ok(response) => response,
            Err(e) => {
                network_error = Some(anyhow::Error::new(e).context("request failed"));
                continue;
            }
        };
        // A release that is not published (yet) is a 404, so try the one before it.
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        if !response.status().is_success() {
            bail!("HTTP {} for the list of {release}", response.status());
        }

        let ids = parse_ids(&response.text()?);
        if ids.is_empty() {
            bail!("the list of {release} contains no probe IDs");
        }
        let list = List { release, ids };
        write_cache(&list);
        return Ok(list);
    }

    match network_error {
        Some(e) => Err(e),
        None => bail!("no list found for {date}"),
    }
}

/// One probe ID per line.
fn parse_ids(text: &str) -> HashSet<u32> {
    text.lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect()
}

fn cache_name(release: &str) -> String {
    format!("inaccurate_probes_{release}.json")
}

fn read_cache(release: &str) -> Option<List> {
    let path = config::cache_path(&cache_name(release))?;
    let cache: Cache = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    (cache.release == release && !cache.ids.is_empty()).then(|| List {
        release: cache.release,
        ids: cache.ids.into_iter().collect(),
    })
}

/// Best effort: a list that cannot be cached is simply fetched again next time.
fn write_cache(list: &List) {
    let Some(path) = config::cache_path(&cache_name(&list.release)) else {
        return;
    };
    let mut ids: Vec<u32> = list.ids.iter().copied().collect();
    ids.sort_unstable();
    let cache = Cache {
        release: list.release.clone(),
        ids,
    };
    if let Ok(bytes) = serde_json::to_vec(&cache) {
        let _ = std::fs::write(path, bytes);
    }
}
