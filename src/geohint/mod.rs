//! Geohint extraction from CHAOS/NSID values.
//!
//! * [`rules`]: the geohint rules, parsed from the Markdown files in `rules/`.
//! * [`index`]: creating an index using the rules and embedded datasets for geohint -> location mappings.
//! * [`text`]: comparing names and splitting a value into tokens.
//! * [`reading`]: the ways a value can be read, and one reading of it.
//! * [`candidates`]: every reading of one value.

#![allow(dead_code, unused_imports)]

mod candidates;
mod index;
mod reading;
mod rules;
mod text;

pub use candidates::candidates;
pub use index::{Index, MIN_POP, Place};
pub use reading::{Hint, Rule};
pub use text::{fold, letter_runs, tokenize};
