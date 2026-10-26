//! Loading and parsing geohint rules from the user-friendly .md files in `../rules`.
//! These files are embedded at build and parsed on first use.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// All rules, parsed from `rules/*.md` (initialized on first use)
pub static RULES: LazyLock<Rules> = LazyLock::new(Rules::load);

/// A geohint that refers to a city (e.g., cloud region or metro code)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CityRef {
    pub city: String,
    pub country: String,
}

/// An IX, the city it is in, and whether it is a 'brand' (i.e., having IXes at multiple cities)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exchange {
    pub place: CityRef,
    /// Present in several cities, this record specifies the home city (e.g., Frankfurt for DECIX)
    pub is_global: bool,
}

/// An operator's own site codes, recognized by its ASN
/// TODO reassess as ASN data is not always included
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operator {
    pub name: String,
    pub asns: Vec<String>,
    /// Records to place mapping for this operator.
    pub sites: HashMap<String, CityRef>,
}

#[derive(Debug, Default)]
pub struct Rules {
    /// Three-letter words that can be misinterpreted as IATA codes (`dns`).
    pub iata_stoplist: HashSet<String>,
    /// IATA codes used that are not official (`FFM` -> `FRA` for Frankfurt).
    pub iata_aliases: HashMap<String, String>,
    /// Words used by operators that can be misinterpreted as city codes (`master`, a city in Iran).
    pub city_stoplist: HashSet<String>,
    /// City aliases used by operators (Londen, Londra, Londres for London).
    pub city_aliases: HashMap<String, String>,
    /// Non-ISO two-letter country tokens mapped to their official ISO (UK -> GB).
    pub country_aliases: HashMap<String, String>,
    /// IX names mapped to its city (LINX -> London).
    pub exchanges: HashMap<String, Exchange>,
    /// Cloud region code to its city (`ap-southeast`).
    pub cloud_regions: HashMap<String, CityRef>,
    /// Metro-area code to its city (`BJS` -> Beijing).
    pub metros: HashMap<String, CityRef>,
    /// Operator-specific rules, based on ASN of IP address.
    pub operators: Vec<Operator>,
}

impl Rules {
    /// Load in rules from .md files
    fn load() -> Rules {
        Rules {
            iata_stoplist: tokens(include_str!("../rules/iata-stoplist.md"))
                .map(|t| t.to_lowercase())
                .collect(),
            iata_aliases: table(include_str!("../rules/iata-aliases.md"), 2)
                .map(|r| (r[0].to_uppercase(), r[1].to_uppercase()))
                .collect(),
            city_stoplist: tokens(include_str!("../rules/city-stoplist.md"))
                .map(|t| t.to_lowercase())
                .collect(),
            city_aliases: table(include_str!("../rules/city-aliases.md"), 2)
                .map(|r| (r[0].to_lowercase(), r[1].clone()))
                .collect(),
            country_aliases: table(include_str!("../rules/country-aliases.md"), 2)
                .map(|r| (r[0].to_uppercase(), r[1].to_uppercase()))
                .collect(),
            exchanges: table(include_str!("../rules/exchanges.md"), 4)
                .map(|r| {
                    let exchange = Exchange {
                        place: CityRef { city: r[1].clone(), country: r[2].clone() },
                        is_global: is_true(&r[3]),
                    };
                    (r[0].to_lowercase(), exchange)
                })
                .collect(),
            cloud_regions: table(include_str!("../rules/cloud-regions.md"), 3)
                .map(|r| (r[0].to_lowercase(), CityRef { city: r[1].clone(), country: r[2].clone() }))
                .collect(),
            metros: table(include_str!("../rules/metros.md"), 3)
                .map(|r| (r[0].to_uppercase(), CityRef { city: r[1].clone(), country: r[2].clone() }))
                .collect(),
            operators: operators(include_str!("../rules/operators.md")),
        }
    }
}

// ---------------------------------------------------------------------------
// Markdown parsing
// ---------------------------------------------------------------------------

/// Parse "token list" files, i.e., white-space separated words.
/// Supports comments (`#`)
fn tokens(md: &str) -> impl Iterator<Item = String> + '_ {
    let mut in_block = false;
    md.lines()
        .filter_map(move |line| {
            if line.trim_start().starts_with("```") {
                in_block = !in_block;
                return None;
            }
            in_block.then(|| line.split('#').next().unwrap_or(""))
        })
        .flat_map(|content| content.split_whitespace().map(str::to_string))
}

/// Parse .md table cells
/// Skips `---|---|` separators
fn row_cells(line: &str) -> Option<Vec<String>> {
    let line = line.trim();
    if !line.starts_with('|') {
        return None;
    }
    let cells: Vec<String> = line
        .trim_matches('|')
        .split('|')
        .map(|c| c.trim().trim_matches('`').trim().to_string())
        .collect();
    // Skip table separators
    let is_separator = cells.iter().all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'));
    (!is_separator).then_some(cells)
}

/// Parse a .md table. Optional columns are skipped (based on the `columns` usize)
fn table(md: &str, columns: usize) -> impl Iterator<Item = Vec<String>> + '_ {
    md.lines().filter_map(row_cells).skip(1).map(move |mut cells| {
        if cells.len() < columns {
            cells.resize(columns, String::new());
        }
        cells
    })
}

/// Whether a yes/no cell is "yes"
/// TODO simply use a single 'y'/'n' value consistently.
fn is_true(cell: &str) -> bool {
    matches!(cell.trim().to_ascii_lowercase().as_str(), "yes" | "true" | "y" | "x" | "✓")
}

/// Parse operator rules (`rules/operators.md`.
/// Each operator rule has its own table and additional sections like `ASNs:` and `## Name`.
fn operators(md: &str) -> Vec<Operator> {
    let mut found: Vec<Operator> = Vec::new();
    let mut past_header = false;

    for line in md.lines() {
        // Parse the first line defining the Operator name
        if let Some(name) = line.trim().strip_prefix("## ") {
            found.push(Operator { name: name.trim().to_string(), asns: Vec::new(), sites: HashMap::new() });
            past_header = false;
            continue;
        }

        let Some(operator) = found.last_mut() else { continue };

        if let Some(rest) = line.trim().trim_start_matches('*').strip_prefix("ASNs:") {
            // Parse ASNs line (`**ASNs:** 25192, 1921`)
            operator.asns.extend(
                rest.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(str::to_string),
            );
        } else if let Some(cells) = row_cells(line) {
            if !std::mem::replace(&mut past_header, true) {
                // First value for the operator mapping (header)
                continue;
            }
            // Parse line containing operator specific geohint to place mapping
            if let [code, city, country, ..] = cells.as_slice() {
                operator
                    .sites
                    .insert(code.to_lowercase(), CityRef { city: city.clone(), country: country.clone() });
            }
        }
    }
    found.retain(|o| !o.asns.is_empty());
    found
}
