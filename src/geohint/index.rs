//! Index built from rules joined with the embedded airports and cities datasets.

use std::collections::HashMap;
use std::io::Cursor;

use anyhow::Result;

use super::rules::{CityRef, RULES, Rules};
use super::text::fold;
use crate::io::{EMBEDDED_AIRPORTS, EMBEDDED_CITIES, decompress_gz, load_airports};
use crate::model::Airport;

/// Avoid small cities that cause noise in city-name detection.
pub const MIN_POP: u32 = 15_000;

/// A place extracted from a geohint. Places may share a location using `key` for airport and city combinations.
/// `coords` `(lat, lon)` will be `None` for a country or unknown city.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    pub key: String,
    pub label: String,
    pub country: String,
    pub coords: Option<(f32, f32)>,
}

/// CHAOS/NSID label geohint to place (airport/city) mapping.
pub struct Index {
    /// IATA code, including metro codes and known aliases (`FFM`, `FRA`, `AMX`).
    codes: HashMap<String, Place>,
    /// City names (at least 5 letters) excluding stoplisted words (`master`) TODO what if cities have less than 5 letters? Lima?
    cities: HashMap<String, Place>,
    /// Aliases of city names, mapped to the standardized city name (`Londres`).
    city_aliases: HashMap<String, String>,
    /// ISO country code (`US`, `NL`).
    countries: HashMap<String, Place>,
    /// Cloud region code (`ap-southeast`)
    regions: HashMap<String, Place>,
    /// IXes (`decix`) mapped to its (default) place and whether it has presence in several cities.
    exchanges: HashMap<String, (Place, bool)>,
    /// Operator specific places (identified by ASN) TODO ASN as unsigned integer rather than String
    operator_codes: HashMap<String, HashMap<String, Place>>,
    /// Three-letter codes often used by operators that happen to be IATA codes (`dns`).
    iata_stoplist: std::collections::HashSet<String>,
    /// Aliases for ISO country codes (`UK` -> `GB`)
    country_aliases: HashMap<String, String>,
}

impl Index {
    /// Load embedded airports, city datasets with `min_pop` filtering
    pub fn embedded() -> Result<Index> {
        let airports = load_airports(Cursor::new(decompress_gz(EMBEDDED_AIRPORTS)?), 0)?;
        let cities = load_airports(Cursor::new(decompress_gz(EMBEDDED_CITIES)?), MIN_POP)?;
        Ok(Index::new(&airports, &cities, &RULES))
    }

    /// `e_airports` and `e_cities` are rows of the embedded datasets (see [`load_airports`]).
    /// `rules` are the markdown derived rules.
    pub fn new(e_airports: &[Airport], e_cities: &[Airport], rules: &Rules) -> Index {
        // Create mapping of geohint labels to embedded Place (track population to keep highest)
        let mut by_citycountry: HashMap<String, (Place, u32)> = HashMap::new(); // london|gb
        let mut by_city: HashMap<String, (Place, u32)> = HashMap::new(); // london
        let mut countries: HashMap<String, Place> = HashMap::new(); // gb

        // Create a mapping for each embedded city based on city name
        for c in e_cities {
            let city_name = fold(&c.city);
            if city_name.is_empty() { // TODO should not exist?
                continue;
            }

            // Create a country mapping for each unique country in the embedded city dataset
            countries.entry(c.country_code.clone()).or_insert_with(|| Place {
                key: format!("country|{}", c.country_code),
                label: c.country_code.clone(),
                country: c.country_code.clone(),
                coords: None, // not a city, so no coordinates
            });

            // City name with country (some city names are used in multiple countries)
            let citycountry_name = format!("{city_name}|{}", c.country_code);

            // Create a mapping for the city itself
            let place = Place {
                key: citycountry_name.clone(),
                label: format!("{}, {}", c.city, c.country_code),
                country: c.country_code.clone(),
                coords: Some((c.lat, c.lon)),
            };

            // Check if this city is already mapped (highest pop within country)
            if by_citycountry.get(&citycountry_name).is_none_or(|(_, pop)| c.pop > *pop) {
                by_citycountry.insert(citycountry_name, (place.clone(), c.pop));
            }

            // Drop short city names and stoplist names
            if city_name.len() < 5 || rules.city_stoplist.contains(&city_name) {
                continue;
            }

            // Store the city itself (highest pop global)
            if by_city.get(&city_name).is_none_or(|(_, pop)| c.pop > *pop) {
                by_city.insert(city_name, (place, c.pop));
            }
        }

        // Map a city reference (named by a rule like IX, metro code, cloud region, ...) to an embedded city
        let resolve_city_ref = |what: &str, place: &CityRef| -> Option<Place> {
            let found = by_citycountry.get(&format!("{}|{}", fold(&place.city), place.country));
            // Warn hints with a city reference that is not in the file (likely typos)
            if found.is_none() {
                eprintln!(
                    "Warning: geohint rule {what} names {}, {}, which is not in the cities dataset; skipped.",
                    place.city, place.country
                );
            }
            found.map(|(found, _)| found.clone())
        };

        // Map airports (using their associated city names, if available)
        let mut iata_codes: HashMap<String, Place> = HashMap::new();
        for a in e_airports {
            // an airport without a city is named after itself
            let base = if a.city.is_empty() { &a.name } else { &a.city };
            iata_codes.entry(a.iata.to_uppercase()).or_insert_with(|| Place {
                key: format!("{}|{}", fold(base), a.country_code),
                label: format!("{base}, {}", a.country_code),
                country: a.country_code.clone(),
                coords: Some((a.lat, a.lon)),
            });
        }
        // Append airports with metro code derived cities
        for (code, city) in &rules.metros {
            if let Some(place) = resolve_city_ref(&format!("metro {code}"), city) {
                iata_codes.entry(code.clone()).or_insert(place);
            }
        }

        // Append for IATA aliases (e.g., `FFM` instead of `FRA`)
        for (alias, code) in &rules.iata_aliases {
            if let Some(place) = iata_codes.get(code).cloned() {
                iata_codes.insert(alias.clone(), place);
            }
        }

        // Map cloud regions to cities
        let regions = rules
            .cloud_regions
            .iter()
            .filter_map(|(region, city)| Some((region.clone(), resolve_city_ref(&format!("cloud region {region}"), city)?)))
            .collect();

        // Map IXes to cities
        let exchanges = rules
            .exchanges
            .iter()
            .filter_map(|(name, x)| {
                Some((name.clone(), (resolve_city_ref(&format!("exchange {name}"), &x.place)?, x.is_global)))
            })
            .collect();

        // Create operator-specific (ASN) mappings of geohint labels to cities
        let mut operator_codes: HashMap<String, HashMap<String, Place>> = HashMap::new();
        for operator in &rules.operators {
            for asn in &operator.asns {
                let known = operator_codes.entry(asn.clone()).or_default();
                for (code, city) in &operator.codes {
                    if let Some(place) = resolve_city_ref(&format!("{} code {code}", operator.name), city) {
                        known.insert(fold(code), place);
                    }
                }
            }
        }

        // Return single `index` containing all parsed and derived rules
        Index {
            codes: iata_codes,
            cities: by_city.into_iter().map(|(name, (place, _))| (name, place)).collect(),
            city_aliases: rules.city_aliases.iter().map(|(from, to)| (fold(from), fold(to))).collect(),
            countries,
            regions,
            exchanges,
            operator_codes,
            iata_stoplist: rules.iata_stoplist.clone(),
            country_aliases: rules.country_aliases.clone(),
        }
    }

    /// Get the place of a three-letter IATA code, skipping stoplist words
    pub fn code(&self, letters: &str) -> Option<&Place> {
        if self.iata_stoplist.contains(&letters.to_lowercase()) {
            return None;
        }
        self.codes.get(&letters.to_uppercase())
    }

    /// Convert a city name into a place (accounting for known alternate spellings)
    pub fn city(&self, text: &str) -> Option<&Place> {
        let name = fold(text);
        self.cities.get(self.city_aliases.get(&name).unwrap_or(&name))
    }

    /// Get the place (country only) from a two-letter code (including non-ISO codes like `uk`)
    pub fn country(&self, token: &str) -> Option<&Place> {
        let code = token.to_uppercase();
        self.countries.get(self.country_aliases.get(&code).unwrap_or(&code))
    }

    /// The ISO country code a two-letter token stands for.
    pub fn iso_code(&self, token: &str) -> String {
        let code = token.to_uppercase();
        self.country_aliases.get(&code).cloned().unwrap_or(code)
    }

    /// Get the city for a region label
    pub fn region(&self, code: &str) -> Option<&Place> {
        self.regions.get(&code.to_lowercase())
    }

    /// The city of an IX, and whether it is present in several cities.
    pub fn exchange(&self, name: &str) -> Option<(&Place, bool)> {
        self.exchanges.get(&name.to_lowercase()).map(|(place, brand)| (place, *brand))
    }

    /// The place an operator (by AS number) means with a code.
    pub fn operator_code(&self, asn: &str, code: &str) -> Option<&Place> {
        self.operator_codes.get(asn)?.get(&fold(code))
    }
}
