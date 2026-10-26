//! Every reading of one CHAOS/NSID value.

use super::index::Index;
use super::reading::Hint;
use super::text::tokenize;

/// Create the possible geohint mappings of a value.
/// A value may have multiple mappings (with conflicting outcomes).
pub fn candidates(value: &str, index: &Index) -> Vec<Hint> {
    let tokens = tokenize(value);
    let mut found = Vec::new();

    
    // TODO get all possible geohint mappings
    // TODO assess based on highest rank
    // TODO assess based on conflicting rules (most agreement wins?)
    // TODO RTT plausibility clause to filter out mappings
    
    found
}
