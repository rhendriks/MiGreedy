//! Every reading of one CHAOS/NSID value.

use super::index::GeoLookup;
use super::reading::{Hint, Rule};
use super::text::{token_split_by_numbers, token_split, tokenize};

/// Create the possible geohint mappings of a value.
/// A value may have multiple mappings (with conflicting outcomes).
pub fn candidates(value: &str, index: &GeoLookup) -> Vec<Hint> {
    let tokens = tokenize(value);
    let mut found = Vec::new();

    // Find IATA tokens for each token
    for pos in 0..tokens.len() {
        read_iata(index, &tokens, pos, &mut found);
    }

    // TODO get all possible geohint mappings
    // TODO assess based on highest rank
    // TODO assess based on conflicting rules (most agreement wins?)
    // TODO RTT plausibility clause to filter out mappings

    found
}

/// IATA codes in the token at `pos`.
///
/// # Arguments
/// * `index` - Contains all known geohint mappings.
/// * `tokens` - List of alphanumeric tokens for the label (to match for IATA codes).
/// * `pos` - Current token being assessed (adjacent tokens are used if useful).
/// * `found` - Geohints found for the current label (append).
///
/// # Example:
/// `ams18-nl`, contains tokens `ams18`, `nl`.
/// The first is an IATA code (mapping to Amsterdam), this is an `IataBare` mapping.
/// However, since the latter token provides a country code, the geohint is upgraded
/// to a more confident `IataCc` mapping.
fn read_iata(index: &GeoLookup, tokens: &[String], pos: usize, found: &mut Vec<Hint>) {
    let token = &tokens[pos];
    match token_split(token) {
        // Check for three letter IATA code
        Some((letters, _number)) if letters.len() == 3 => {
            // Extract place from possible IATA code
            let Some(place) = index.iata_code(letters) else { return };
            // Check for neighboring country codes (can provide stronger confidence)
            let next_to_own_country = [pos.checked_sub(1), Some(pos + 1)]
                .into_iter()
                .flatten()
                .filter_map(|i| tokens.get(i))
                .any(|neighbour| index.iso_code(neighbour) == place.country);
            // Add extracted place as IataBare or IataCc (if there is a neighboring country code)
            let rule = if next_to_own_country { Rule::IataCc } else { Rule::IataBare };
            found.push(Hint { place: place.clone(), rule, token: token.clone() });
        }
        // TODO will fail when a code is glued with additional text (`seana`, `atlus`)
        Some(_) => {}
        None => {
            // Try to split as words separated by numbers
            for run in token_split_by_numbers(token).into_iter().filter(|run| run.len() == 3) {
                if let Some(place) = index.iata_code(run) {
                    found.push(Hint { place: place.clone(), rule: Rule::IataGlued, token: run.to_string() });
                }
                // TODO will not find IataCc for e.g., `ams3nl`
            }
        }
    }
}
