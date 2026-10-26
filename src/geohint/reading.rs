//! The ways a value can be read, and one reading of it.
//! The index might output multiple mappings for a single value.
//! This tracks such a mapping alongside the rule that identified it.

use super::index::Place;

/// How a place was read from a value.
/// The variants are ordered by rank: [`Rule::rank`] is lower for a stronger reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule { // TODO assess order
    /// Operator-specific codes.
    OperatorCode,
    /// Cloud region: `ap-southeast-1`.
    Region, // TODO merge into .md for operator-specific codes?
    /// UN/LOCODE: a country code and a location code in one token, `nlams`, `uschi`.
    Locode,
    /// IATA code next to its own country: `lhr-gb`.
    IataCc,
    /// Internet exchange that is in one city: `dns-nixsk-01`.
    Ix,
    /// The name of a city spelled out, in one word or several: `amsterdam1`, `sao_paulo`.
    CityName,
    /// IATA code that is the whole token, with or without a server number: `ams`, `ams18`.
    IataBare,
    /// IATA code glued to other characters in one token: `seana1`, `adns1v1man`.
    IataGlued,
    /// Exchange present in several cities: `decix`.
    IxGlobal,
    /// Two-letter country token: `us39-12`.
    Country,
}

impl Rule {
    /// The rank of a reading.
    pub fn rank(self) -> u8 { // TODO assess ranking
        match self {
            Rule::Locode | Rule::Region => 0,
            Rule::IataCc | Rule::Ix | Rule::OperatorCode => 1,
            Rule::CityName => 2,
            Rule::IataBare => 3,
            Rule::IataGlued | Rule::IxGlobal => 4,
            Rule::Country => 5,
        }
    }
}

/// A geohint mapping for a token: containing the mapped place, the rule that found it, and the token itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Hint {
    pub place: Place,
    pub rule: Rule,
    pub token: String,
}
