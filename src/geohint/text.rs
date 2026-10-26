//! Text helpers: comparing names and splitting a value into tokens.

use unicode_normalization::UnicodeNormalization;

/// Sanitize a CHAOS/NSID value into a token without accents, uppercase, and non-alphanumeric symbols.
pub fn fold(text: &str) -> String {
    text.nfd()
        .filter(|c| !('\u{0300}'..='\u{036f}').contains(c))
        .flat_map(char::to_lowercase)
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

/// Create multiple tokens from a value: `Dns-LHR18.example` -> `dns`, `lhr18`, `example`.
pub fn tokenize(value: &str) -> Vec<String> {
    value
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

/// Find string tokens with numbers as separators: `a2lhr` has `a` and `lhr`.
pub fn letter_runs(token: &str) -> Vec<&str> {
    token.split(|c: char| !c.is_ascii_lowercase()).filter(|run| !run.is_empty()).collect()
}
