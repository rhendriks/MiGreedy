//! Geohint extraction from CHAOS/NSID values.
//!
//! Such records may contain hints like IATA airport codes (`lhr18`),
//! a UN/LOCODE (`nlams`), a city (`amsterdam17`), an IX (`dns-decix01`),
//! a cloud region (`ap-southeast-1`), or a country/US state (`us39-12`, `edge01-ny`).
//!
//! This file contains rules that turn these values into places.
//! We also provide operator-specific rules based on feedback/ground-truth.
//!
//! **DISCLAIMER:**
//!
//! This code is best-effort.
//! There are many exceptions where this might fail.
//! This can lead to false inferences of locations, or failures to recognize one.
//! We tested these rules on large datasets of collected CHAOS/NSID values
//! from our daily LACeS pipeline. This shows good coverage.
//! However, manual (human) verification is not always feasible.
//! We appreciate pull requests for missing/wrong rules :)

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use unicode_normalization::UnicodeNormalization;

/// Words common in server names that are also IATA codes.
/// A three-letter token is only read as an airport when it is not one of these.
static CODE_STOPLIST: LazyLock<HashSet<String>> =
    LazyLock::new(|| parse_tokens(include_str!("../rules/iata-stoplist.md")).collect());

/// Words common in server names that are also city names.
static CITY_STOPLIST: LazyLock<HashSet<String>> =
    LazyLock::new(|| parse_tokens(include_str!("../rules/city-stoplist.md")).collect());

/// US state two-letter codes (uppercased).
static US_STATES: LazyLock<HashSet<String>> = LazyLock::new(|| {
    parse_tokens(include_str!("../rules/us-states.md"))
        .map(|s| s.to_uppercase())
        .collect()
});

/// Non-standardized IATA codes used by operators mapped to the real IATA codes.
static CODE_ALIASES: LazyLock<Vec<(String, String)>> = LazyLock::new(|| {
    parse_table(include_str!("../rules/iata-aliases.md"), 2)
        .map(|r| (r[0].to_uppercase(), r[1].to_uppercase()))
        .collect()
});

/// Two-letter country tokens that are not ISO 3166 (uk-lon), mapped to the ISO code (uppercased).
static COUNTRY_ALIAS_MAP: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    parse_table(include_str!("../rules/country-aliases.md"), 2)
        .map(|r| (r[0].to_uppercase(), r[1].to_uppercase()))
        .collect()
});

/// City names used by operators that are not the standardized/English name.
static CITY_ALIAS_MAP: LazyLock<HashMap<String, String>> = LazyLock::new(|| {
    parse_table(include_str!("../rules/city-aliases.md"), 2)
        .map(|r| (r[0].clone(), r[1].clone()))
        .collect()
});

/// `(name, city, country, is_brand)` for IXes.
/// `is_brand` marks an IX with exchanges in several cities,
/// its home city is when nothing else in the value names a place.
static EXCHANGES: LazyLock<Vec<(String, String, String, bool)>> = LazyLock::new(|| {
    parse_table(include_str!("../rules/exchanges.md"), 4)
        .map(|r| (r[0].clone(), r[1].clone(), r[2].clone(), is_truthy(&r[3])))
        .collect()
});

/// Cloud region codes to their `(region, city, country)`.
static CLOUD_REGIONS: LazyLock<Vec<(String, String, String)>> = LazyLock::new(|| {
    parse_table(include_str!("../rules/cloud-regions.md"), 3)
        .map(|r| (r[0].clone(), r[1].clone(), r[2].clone()))
        .collect()
});

/// Operator-specific site codes, applied only to a prefix announced by that operator's AS.
/// TODO apply at IP address granularity instead
static OPERATORS: LazyLock<Vec<Operator>> =
    LazyLock::new(|| parse_operators(include_str!("../rules/operators.md")));

/// Metro-area `(code, city, country)` codes.
static METROS: LazyLock<Vec<(String, String, String)>> = LazyLock::new(|| {
    parse_table(include_str!("../rules/metros.md"), 3)
        .map(|r| (r[0].to_uppercase(), r[1].clone(), r[2].clone()))
        .collect()
});

/// Parse a "token list" rule file.
/// Whitespace-separated lowercase tokens.
/// `#` lines are treated as comments.
fn parse_tokens(md: &str) -> impl Iterator<Item = String> + '_ {
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

/// Parse a "table" rule file.
/// Markdown tables format.
fn parse_table(md: &str, columns: usize) -> impl Iterator<Item = Vec<String>> + '_ {
    let mut seen_header = false;
    md.lines().filter_map(move |line| {
        let line = line.trim();
        if !line.starts_with('|') {
            return None;
        }
        let mut cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().trim_matches('`').trim().to_string())
            .collect();
        // a separator row like |---|:--:|
        if cells
            .iter()
            .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            return None;
        }
        if !seen_header {
            seen_header = true;
            return None;
        }
        if cells.len() < columns {
            cells.resize(columns, String::new());
        }
        Some(cells)
    })
}

/// Whether a `is_true` value is set (`yes`, `true`, `y`, `x`, `✓`).
/// TODO use a single token consistently
fn is_truthy(cell: &str) -> bool {
    matches!(
        cell.trim().to_ascii_lowercase().as_str(),
        "yes" | "true" | "y" | "x" | "✓"
    )
}

/// Parse the operator rule file.
/// One ruleset per operator that maps codes to places.
fn parse_operators(md: &str) -> Vec<Operator> {
    let mut operators: Vec<Operator> = Vec::new();
    let mut seen_header = false;
    for line in md.lines() {
        let line = line.trim();
        if line.starts_with("## ") {
            operators.push(Operator {
                asns: Vec::new(), // TODO match using ip address rather than ASN
                sites: Vec::new(),
            });
            seen_header = false;
            continue;
        }
        let Some(op) = operators.last_mut() else {
            continue;
        };
        // the ASNs line (allowing markdown emphasis like **ASNs:**); its digit runs are the ASes
        if line.trim_matches('*').to_ascii_lowercase().starts_with("asns") {
            op.asns.extend(
                line.split(|c: char| !c.is_ascii_digit())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            );
            continue;
        }
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().trim_matches('`').trim().to_string())
            .collect();
        if cells
            .iter()
            .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue; // separator row
        }
        if !seen_header {
            seen_header = true;
            continue; // header row
        }
        if cells.len() >= 3 && !cells[0].is_empty() {
            op.sites
                .push((cells[0].clone(), (cells[1].clone(), cells[2].clone())));
        }
    }
    operators.retain(|o| !o.asns.is_empty());
    operators
}

// ---------------------------------------------------------------------------
// Rules and places
// ---------------------------------------------------------------------------

/// Reading rules ranked by preference
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rule {
    Locode,
    Region,
    IataCc,
    IataNum,
    Exchange,
    Site,
    City,
    IataBare,
    IataGlued,
    Brand,
    Country,
    State,
}

impl Rule {
    /// Rule rankings (lowest preferred)
    pub fn rank(self) -> u8 {
        use Rule::*;
        match self {
            Locode | Region => 0,
            IataCc | IataNum | Exchange | Site => 1,
            City => 2,
            IataBare => 3,
            IataGlued | Brand => 4,
            Country | State => 5,
        }
    }

    /// Whether this is one of the `iata-*` rules.
    pub fn is_iata(self) -> bool {
        matches!(self, Rule::IataCc | Rule::IataNum | Rule::IataBare | Rule::IataGlued)
    }

    /// A state and a country from the same two letters are graded together (kind() in Python).
    pub fn kind(self) -> Rule {
        if matches!(self, Rule::State) {
            Rule::Country
        } else {
            self
        }
    }
}

/// A place the rules can name. Places with the same `key` are one location (an airport and its
/// city's name count once). `lat`/`lon` are `None` for a country or US state, and for a place
/// whose reference data lacks coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub key: String,
    pub label: String,
    pub country: String,
    pub lat: Option<f32>,
    pub lon: Option<f32>,
}

/// One reading of one value.
#[derive(Clone, Debug)]
pub struct Hint {
    pub place: Place,
    pub rule: Rule,
    pub token: String,
    pub pos: i32,
    /// A two-letter country token next to the code that is not the code's country (dns-cal-us).
    pub contra: Option<String>,
    pub contra_pos: Option<i32>,
}

impl Hint {
    pub fn country(&self) -> &str {
        &self.place.country
    }
}

// ---------------------------------------------------------------------------
// Index of reference places
// ---------------------------------------------------------------------------

/// One airport row for [`Index::build`]: `(iata, name, city, country, lat, lon)`.
pub struct AirportRow<'a> {
    pub iata: &'a str,
    pub name: &'a str,
    pub city: &'a str,
    pub country: &'a str,
    pub lat: Option<f32>,
    pub lon: Option<f32>,
}

/// One metro row: a code that resolves to a city but has no coordinates of its own.
pub struct MetroRow<'a> {
    pub iata: &'a str,
    pub city: &'a str,
    pub country: &'a str,
}

/// One city row: `(name, country, lat, lon, pop)`.
pub struct CityRow<'a> {
    pub name: &'a str,
    pub country: &'a str,
    pub lat: Option<f32>,
    pub lon: Option<f32>,
    pub pop: u32,
}

/// An operator's known site names, by the ASes that answer for it.
pub struct Operator {
    pub asns: Vec<String>,
    /// `(site name, (city, country))`.
    pub sites: Vec<(String, (String, String))>,
}

/// The places the rules can name.
pub struct Index {
    /// IATA (and alias) code, uppercased, to its place.
    codes: HashMap<String, Place>,
    /// Folded city name to its place (length >= 5, not stoplisted).
    cities: HashMap<String, Place>,
    /// Country code to its place.
    countries: HashMap<String, Place>,
    /// US state code to its place.
    states: HashMap<String, Place>,
    /// Cloud region code to its place.
    regions: HashMap<String, Place>,
    /// Exchange / colo name to its place and whether it is a brand.
    exchanges: HashMap<String, (Place, Rule)>,
    /// ASN to {folded operator site name: place}.
    sites: HashMap<String, HashMap<String, Place>>,
}

impl Index {
    pub fn build(
        airports: &[AirportRow],
        metros: &[MetroRow],
        cities: &[CityRow],
        operators: &[Operator],
    ) -> Index {
        // the most populous city of a name (per country, and overall) wins
        let mut by_city: HashMap<String, Place> = HashMap::new();
        let mut by_city_pop: HashMap<String, u32> = HashMap::new();
        let mut countries: HashMap<String, Place> = HashMap::new();
        let mut city_idx: HashMap<String, Place> = HashMap::new();
        let mut city_pop: HashMap<String, u32> = HashMap::new();

        for c in cities {
            let ck = format!("{}|{}", fold(c.name), c.country);
            if by_city_pop.get(&ck).is_none_or(|&p| c.pop > p) {
                by_city.insert(
                    ck.clone(),
                    Place {
                        key: ck.clone(),
                        label: format!("{}, {}", c.name, c.country),
                        country: c.country.to_string(),
                        lat: c.lat,
                        lon: c.lon,
                    },
                );
                by_city_pop.insert(ck.clone(), c.pop);
            }
            countries.entry(c.country.to_string()).or_insert_with(|| Place {
                key: format!("country|{}", c.country),
                label: c.country.to_string(),
                country: c.country.to_string(),
                lat: None,
                lon: None,
            });
            let k = fold(c.name);
            if k.len() < 5 || CITY_STOPLIST.contains(k.as_str()) {
                continue;
            }
            if city_pop.get(&k).is_none_or(|&p| c.pop > p) {
                city_idx.insert(
                    k.clone(),
                    Place {
                        key: format!("{}|{}", k, c.country),
                        label: format!("{}, {}", c.name, c.country),
                        country: c.country.to_string(),
                        lat: c.lat,
                        lon: c.lon,
                    },
                );
                city_pop.insert(k, c.pop);
            }
        }

        // a place without coordinates of its own (a metro code, a cloud region) is at its city
        let at_city = |city: &str, country: &str| -> Place {
            let key = format!("{}|{}", fold(city), country);
            let (lat, lon) = by_city.get(&key).map_or((None, None), |p| (p.lat, p.lon));
            Place {
                key,
                label: format!("{}, {}", city, country),
                country: country.to_string(),
                lat,
                lon,
            }
        };

        let mut codes: HashMap<String, Place> = HashMap::new();
        for a in airports {
            let base = if a.city.is_empty() { a.name } else { a.city };
            codes.insert(
                a.iata.to_uppercase(),
                Place {
                    key: format!("{}|{}", fold(base), a.country),
                    label: format!("{}, {}", base, a.country),
                    country: a.country.to_string(),
                    lat: a.lat,
                    lon: a.lon,
                },
            );
        }
        // a metro code never shadows an airport of the same code (BER is both)
        for m in metros {
            codes
                .entry(m.iata.to_uppercase())
                .or_insert_with(|| at_city(m.city, m.country));
        }
        for (alias, code) in CODE_ALIASES.iter() {
            if let Some(place) = codes.get(code) {
                codes.insert(alias.to_string(), place.clone());
            }
        }

        let states = US_STATES
            .iter()
            .map(|s| {
                (
                    s.to_string(),
                    Place {
                        key: format!("state|US-{s}"),
                        label: format!("{s}, US"),
                        country: "US".to_string(),
                        lat: None,
                        lon: None,
                    },
                )
            })
            .collect();

        let regions = CLOUD_REGIONS
            .iter()
            .map(|(region, city, country)| (region.to_string(), at_city(city, country)))
            .collect();

        // a name whose city is not in the cities list names nothing
        let mut exchanges: HashMap<String, (Place, Rule)> = HashMap::new();
        for (name, city, country, brand) in EXCHANGES.iter() {
            if let Some(place) = by_city.get(&format!("{}|{}", fold(city), country)) {
                let rule = if *brand { Rule::Brand } else { Rule::Exchange };
                exchanges.insert(name.to_string(), (place.clone(), rule));
            }
        }

        // the site names operators give their servers that the rules would misread, by AS
        let mut sites: HashMap<String, HashMap<String, Place>> = HashMap::new();
        for operator in operators {
            for asn in &operator.asns {
                let known = sites.entry(asn.clone()).or_default();
                for (name, (city, country)) in &operator.sites {
                    if let Some(place) = by_city.get(&format!("{}|{}", fold(city), country)) {
                        known.insert(fold(name), place.clone());
                    }
                }
            }
        }

        Index {
            codes,
            cities: city_idx,
            countries,
            states,
            regions,
            exchanges,
            sites,
        }
    }

    /// The place of a three-letter code, unless the token is a stoplisted word.
    fn code(&self, letters: &str) -> Option<Place> {
        if CODE_STOPLIST.contains(letters) {
            None
        } else {
            self.codes.get(&letters.to_uppercase()).cloned()
        }
    }

    /// The place of an uppercased code, bypassing the stoplist (used for LOCODEs).
    fn raw_code(&self, upper: &str) -> Option<&Place> {
        self.codes.get(upper)
    }

    /// The place of a city name (its alias, if any).
    fn city(&self, text: &str) -> Option<Place> {
        let k = fold(text);
        let k = CITY_ALIAS_MAP.get(k.as_str()).map_or(k.clone(), |a| fold(a));
        self.cities.get(&k).cloned()
    }

    fn is_country(&self, token: &str) -> bool {
        self.countries.contains_key(&country_of(token))
    }
}

// ---------------------------------------------------------------------------
// Text helpers (the hand-rolled replacements for the Python regexes)
// ---------------------------------------------------------------------------

/// Fold text to a comparable key: NFD-normalize, drop combining marks, lowercase, keep `[a-z0-9]`.
pub fn fold(text: &str) -> String {
    text.nfd()
        .filter(|&c| !('\u{0300}'..='\u{036f}').contains(&c))
        .flat_map(char::to_lowercase)
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

/// Split a value into lowercased `[a-z0-9]` tokens (`TOKEN_SPLIT`).
fn split(value: &str) -> Vec<String> {
    value
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// Map a two-letter token to its country code, applying the non-ISO aliases.
fn country_of(token: &str) -> String {
    let cc = token.to_uppercase();
    COUNTRY_ALIAS_MAP
        .get(cc.as_str())
        .map_or(cc, |a| a.to_string())
}

/// A token of letters then an optional server number (`LETTERS_DIGITS` fullmatch): returns
/// `(letters, digits)`. `None` when the token is not of that shape.
fn letters_digits(token: &str) -> Option<(&str, &str)> {
    let bytes = token.as_bytes();
    let letters = bytes.iter().take_while(|b| b.is_ascii_lowercase()).count();
    if letters == 0 {
        return None;
    }
    if bytes[letters..].iter().all(u8::is_ascii_digit) {
        Some((&token[..letters], &token[letters..]))
    } else {
        None
    }
}

/// The maximal runs of ascii letters in a token (`re.findall('[a-z]+')`).
fn alpha_runs(token: &str) -> Vec<&str> {
    token
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Whether the token is all letters, length >= 2 (`[a-z]{2,}`).
fn is_word(token: &str) -> bool {
    token.len() >= 2 && token.bytes().all(|b| b.is_ascii_lowercase())
}

/// A name with any trailing server number removed (`re.sub(r'\d+$', '', name)`).
fn strip_trailing_digits(name: &str) -> &str {
    name.trim_end_matches(|c: char| c.is_ascii_digit())
}

/// Whether a value is a binary NSID (`BINARY`): escaped `\xNN` bytes or control characters.
fn is_binary(value: &str) -> bool {
    if value
        .chars()
        .any(|c| (c as u32) <= 0x1f || c == '\u{7f}' || c == '\u{fffd}')
    {
        return true;
    }
    value.as_bytes().windows(4).any(|w| {
        w[0] == b'\\'
            && w[1].eq_ignore_ascii_case(&b'x')
            && w[2].is_ascii_hexdigit()
            && w[3].is_ascii_hexdigit()
    })
}

/// The cloud-region codes present in a lowercased value (`CLOUD_REGION_RE`). A region is bounded
/// by a non-letter before and a non-digit after; unknown regions name nothing, so only the known
/// region codes are searched for.
fn find_cloud_regions(lower: &str) -> Vec<String> {
    let bytes = lower.as_bytes();
    let mut out = Vec::new();
    for (region, _, _) in CLOUD_REGIONS.iter() {
        let region = region.as_str();
        let mut from = 0;
        while let Some(rel) = lower[from..].find(region) {
            let start = from + rel;
            let end = start + region.len();
            let before_ok = start == 0 || !bytes[start - 1].is_ascii_lowercase();
            let after_ok = end >= bytes.len() || !bytes[end].is_ascii_digit();
            if before_ok && after_ok {
                out.push(region.to_string());
            }
            from = start + 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// candidates(): every reading of one value
// ---------------------------------------------------------------------------

fn push(
    found: &mut Vec<Hint>,
    place: Option<Place>,
    rule: Rule,
    token: &str,
    pos: i32,
    contra: Option<String>,
    contra_pos: Option<i32>,
) {
    if let Some(place) = place {
        found.push(Hint {
            place,
            rule,
            token: token.to_string(),
            pos,
            contra,
            contra_pos,
        });
    }
}

/// A two-letter token read as a country and (if applicable) a US state.
fn two_letter(found: &mut Vec<Hint>, index: &Index, letters: &str, pos: i32) {
    push(
        found,
        index.countries.get(&country_of(letters)).cloned(),
        Rule::Country,
        letters,
        pos,
        None,
        None,
    );
    let upper = letters.to_uppercase();
    if US_STATES.contains(upper.as_str()) {
        push(
            found,
            index.states.get(&upper).cloned(),
            Rule::State,
            letters,
            pos,
            None,
            None,
        );
    }
}

/// The key under which `name` (or its number-stripped form) appears in `names`, if any.
fn site_name<V>(names: &HashMap<String, V>, name: &str) -> Option<String> {
    if names.contains_key(name) {
        return Some(name.to_string());
    }
    let stripped = strip_trailing_digits(name);
    if stripped != name && names.contains_key(stripped) {
        return Some(stripped.to_string());
    }
    None
}

/// Every reading of one value, whatever its rule. `asns` are the ASes of the prefix that
/// answered, whose operators' site names apply.
pub fn candidates(value: &str, index: &Index, asns: &[String]) -> Vec<Hint> {
    if is_binary(value) {
        return Vec::new();
    }
    let lower = value.to_lowercase();
    let tokens = split(value);
    let mut found: Vec<Hint> = Vec::new();

    // cloud regions (pos is -1, as in the Python)
    for region in find_cloud_regions(&lower) {
        push(&mut found, index.regions.get(&region).cloned(), Rule::Region, &region, -1, None, None);
    }

    let operator_sites: Vec<&HashMap<String, Place>> =
        asns.iter().filter_map(|asn| index.sites.get(asn)).collect();

    // exchange, colo and operator site names, with a server number or not, in one token or two
    // words (ams-ix, nix-sk; not m-ix or any-2)
    let mut site_tokens: HashSet<i32> = HashSet::new();
    for i in 0..tokens.len() {
        let pos = i as i32;
        let joined = is_word(&tokens[i])
            && i + 1 < tokens.len()
            && letters_digits(&tokens[i + 1]).is_some();
        let names: Vec<(String, usize)> = if joined {
            vec![
                (tokens[i].clone(), 1),
                (format!("{}{}", tokens[i], tokens[i + 1]), 2),
            ]
        } else {
            vec![(tokens[i].clone(), 1)]
        };
        for (name, span) in names {
            let mut matched = false;
            if let Some(key) = site_name(&index.exchanges, &name) {
                let (place, rule) = index.exchanges[&key].clone();
                push(&mut found, Some(place), rule, &key, pos, None, None);
                matched = true;
            }
            for names in &operator_sites {
                if let Some(key) = site_name(names, &name) {
                    push(&mut found, names.get(&key).cloned(), Rule::Site, &key, pos, None, None);
                    matched = true;
                }
            }
            if matched {
                for t in i..i + span {
                    site_tokens.insert(t as i32);
                }
            }
        }
    }

    for i in 0..tokens.len() {
        let pos = i as i32;
        let token = &tokens[i];
        // letters, then an optional server number: ams5, lhr18, uschi, usmia201, amsterdam1
        let Some((alpha, digits)) = letters_digits(token) else {
            // letters and digits mixed: nsa2amsterdam, adns1v1man, ny00vmp. A city in it counts
            // as is; a code or a leading two-letter country/state only with the scheme behind it
            for (r, run) in alpha_runs(token).into_iter().enumerate() {
                if run.len() == 3 {
                    push(&mut found, index.code(run), Rule::IataGlued, run, pos, None, None);
                }
                if run.len() >= 5 {
                    push(&mut found, index.city(run), Rule::City, run, pos, None, None);
                }
                if run.len() == 2 && r == 0 && token.starts_with(run) {
                    two_letter(&mut found, index, run, pos);
                }
            }
            continue;
        };
        let has_digits = !digits.is_empty();
        if alpha.len() == 2 {
            two_letter(&mut found, index, alpha, pos);
        } else if alpha.len() == 3 {
            if let Some(place) = index.code(alpha) {
                let neighbours = [
                    if i > 0 { country_of(&tokens[i - 1]) } else { String::new() },
                    if i + 1 < tokens.len() { country_of(&tokens[i + 1]) } else { String::new() },
                ];
                if neighbours.iter().any(|n| *n == place.country) {
                    push(&mut found, Some(place), Rule::IataCc, token, pos, None, None);
                } else {
                    // next to another country's code (dns-cal-us is not Campbeltown, GB), unless
                    // those two letters turn out to be the operator's (mu-lon-vlt-02)
                    match neighbours.iter().position(|cc| index.countries.contains_key(cc)) {
                        None => {
                            let rule = if has_digits { Rule::IataNum } else { Rule::IataBare };
                            push(&mut found, Some(place), rule, token, pos, None, None);
                        }
                        Some(side) => {
                            let contra_pos = if side == 0 { pos - 1 } else { pos + 1 };
                            let contra = neighbours[side].clone();
                            push(&mut found, Some(place), Rule::IataBare, token, pos, Some(contra), Some(contra_pos));
                        }
                    }
                }
            }
        } else {
            let mut named = false;
            if alpha.len() == 5 {
                // UN/LOCODE: the country must be the code's country (nlams, jptyo, uschi)
                let code = alpha[2..].to_uppercase();
                if let Some(place) = index.raw_code(&code) {
                    if place.country == alpha[..2].to_uppercase() {
                        push(&mut found, Some(place.clone()), Rule::Locode, token, pos, None, None);
                        named = true;
                    }
                }
            }
            if alpha.len() >= 5 {
                if let Some(city) = index.city(alpha) {
                    push(&mut found, Some(city), Rule::City, token, pos, None, None);
                    named = true;
                }
            }
            // a code with a suffix before the server number: seana1, bomna1, gruna3
            if !named && has_digits && alpha.len() <= 6 && alpha.len() >= 3 {
                push(&mut found, index.code(&alpha[..3]), Rule::IataGlued, &alpha[..3], pos, None, None);
            }
        }
    }

    // city names of two or three words: sao_paulo, new-york, frankfurt_am_main
    for n in 2..=3usize {
        if tokens.len() < n {
            continue;
        }
        for i in 0..=tokens.len() - n {
            let words = &tokens[i..i + n];
            if !words.iter().all(|w| w.bytes().all(|b| b.is_ascii_lowercase())) {
                continue;
            }
            let pos = i as i32;
            let joined: String = words.concat();
            let token = words.join("_");
            push(&mut found, index.city(&joined), Rule::City, &token, pos, None, None);
        }
    }

    // a code inside an exchange or site name is that name's (ams in ams-ix, vix)
    found
        .into_iter()
        .filter(|h| !(h.rule.is_iata() && site_tokens.contains(&h.pos)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_index() -> Index {
        let airports = [
            AirportRow { iata: "LHR", name: "London Heathrow", city: "London", country: "GB", lat: Some(51.47), lon: Some(-0.45) },
            AirportRow { iata: "FRA", name: "Frankfurt", city: "Frankfurt am Main", country: "DE", lat: Some(50.03), lon: Some(8.57) },
            AirportRow { iata: "AMS", name: "Schiphol", city: "Amsterdam", country: "NL", lat: Some(52.31), lon: Some(4.76) },
            AirportRow { iata: "CHI", name: "Chicago", city: "Chicago", country: "US", lat: Some(41.98), lon: Some(-87.9) },
        ];
        let cities = [
            CityRow { name: "London", country: "GB", lat: Some(51.5), lon: Some(-0.12), pop: 8_900_000 },
            CityRow { name: "Frankfurt am Main", country: "DE", lat: Some(50.11), lon: Some(8.68), pop: 753_000 },
            CityRow { name: "Amsterdam", country: "NL", lat: Some(52.37), lon: Some(4.9), pop: 820_000 },
            CityRow { name: "Chicago", country: "US", lat: Some(41.85), lon: Some(-87.65), pop: 2_700_000 },
            CityRow { name: "Singapore", country: "SG", lat: Some(1.29), lon: Some(103.85), pop: 5_600_000 },
            CityRow { name: "New York", country: "US", lat: Some(40.71), lon: Some(-74.0), pop: 8_300_000 },
            CityRow { name: "Sao Paulo", country: "BR", lat: Some(-23.55), lon: Some(-46.63), pop: 12_300_000 },
        ];
        Index::build(&airports, &[], &cities, &[])
    }

    /// The single strongest reading's (rule, place label), for terse assertions.
    fn best(value: &str, index: &Index) -> Option<(Rule, String)> {
        best_with(value, index, &[])
    }

    /// Like [`best`], but for a prefix announced by the given ASes (operator site rules apply).
    fn best_with(value: &str, index: &Index, asns: &[String]) -> Option<(Rule, String)> {
        candidates(value, index, asns)
            .into_iter()
            .min_by_key(|h| h.rule.rank())
            .map(|h| (h.rule, h.place.label))
    }

    #[test]
    fn operator_site_rule_overrides_iata() {
        // RcodeZero (AS1921) names its Seattle site `stl`; the generic rules read `stl` as the
        // St Louis airport (STL). The operator rule must win, but only for that AS.
        let airports = [AirportRow {
            iata: "STL",
            name: "St Louis",
            city: "St Louis",
            country: "US",
            lat: Some(38.75),
            lon: Some(-90.37),
        }];
        let cities = [
            CityRow { name: "Seattle", country: "US", lat: Some(47.6), lon: Some(-122.3), pop: 750_000 },
            CityRow { name: "St Louis", country: "US", lat: Some(38.63), lon: Some(-90.2), pop: 300_000 },
        ];
        let idx = Index::build(&airports, &[], &cities, &OPERATORS);
        assert_eq!(
            best_with("dns-stl-01", &idx, &["1921".to_string()]),
            Some((Rule::Site, "Seattle, US".to_string()))
        );
        assert_eq!(
            best("dns-stl-01", &idx),
            Some((Rule::IataBare, "St Louis, US".to_string()))
        );
    }

    #[test]
    fn operators_file_loads() {
        // the embedded rules/operators.md parses into the two documented operators
        assert_eq!(OPERATORS.len(), 2);
        assert!(OPERATORS.iter().any(|o| o.asns == ["25192"]));
        assert!(OPERATORS.iter().any(|o| o.asns == ["1921"] && o.sites.len() == 68));
    }

    #[test]
    fn metros_file_loads() {
        assert_eq!(METROS.len(), 23);
        assert!(
            METROS
                .iter()
                .any(|(code, city, cc)| code == "ASH" && city == "Ashburn" && cc == "US")
        );
    }

    #[test]
    fn metro_code_resolves() {
        // ASH is not an airport; the metro table lets `ash1` resolve to Ashburn.
        let cities = [CityRow {
            name: "Ashburn",
            country: "US",
            lat: Some(39.04),
            lon: Some(-77.49),
            pop: 50_000,
        }];
        let metros: Vec<MetroRow> = METROS
            .iter()
            .map(|(code, city, cc)| MetroRow {
                iata: code.as_str(),
                city: city.as_str(),
                country: cc.as_str(),
            })
            .collect();
        let idx = Index::build(&[], &metros, &cities, &[]);
        assert_eq!(
            best("ash1", &idx),
            Some((Rule::IataNum, "Ashburn, US".to_string()))
        );
    }

    #[test]
    fn iata_with_server_number() {
        let idx = test_index();
        assert_eq!(best("lhr18", &idx), Some((Rule::IataNum, "London, GB".into())));
    }

    #[test]
    fn iata_next_to_its_country() {
        let idx = test_index();
        assert_eq!(best("lhr-gb", &idx), Some((Rule::IataCc, "London, GB".into())));
    }

    #[test]
    fn locode() {
        let idx = test_index();
        assert_eq!(best("nlams", &idx), Some((Rule::Locode, "Amsterdam, NL".into())));
        assert_eq!(best("uschi", &idx), Some((Rule::Locode, "Chicago, US".into())));
    }

    #[test]
    fn city_name() {
        let idx = test_index();
        assert_eq!(best("amsterdam1", &idx), Some((Rule::City, "Amsterdam, NL".into())));
    }

    #[test]
    fn two_word_city() {
        let idx = test_index();
        assert_eq!(best("new-york", &idx), Some((Rule::City, "New York, US".into())));
        assert_eq!(best("sao_paulo", &idx), Some((Rule::City, "Sao Paulo, BR".into())));
    }

    #[test]
    fn cloud_region() {
        let idx = test_index();
        assert_eq!(best("ap-southeast-1", &idx), Some((Rule::Region, "Singapore, SG".into())));
    }

    #[test]
    fn exchange_brand() {
        let idx = test_index();
        // 'dns' is stoplisted, so the only reading is the DE-CIX brand -> Frankfurt
        assert_eq!(best("dns-decix-dns01", &idx), Some((Rule::Brand, "Frankfurt am Main, DE".into())));
    }

    #[test]
    fn stoplisted_word_names_nothing() {
        let idx = test_index();
        assert!(candidates("dns", &idx, &[]).is_empty());
    }

    #[test]
    fn binary_nsid_names_nothing() {
        let idx = test_index();
        assert!(candidates("ns1\\x00\\xff", &idx, &[]).is_empty());
        assert!(candidates("a\u{0001}b", &idx, &[]).is_empty());
    }

    #[test]
    fn opaque_token_names_nothing() {
        let idx = test_index();
        assert!(candidates("d891d64c2d3318", &idx, &[]).is_empty());
    }

    #[test]
    fn us_state_in_mixed_token() {
        let idx = test_index();
        let hints = candidates("ny00vmp", &idx, &[]);
        assert!(hints.iter().any(|h| h.rule == Rule::State && h.place.label == "NY, US"));
    }
}
