//! Local configuration: storing and resolving the RIPE Atlas API key.
//!
//! Scheduling measurements (`--measure`) needs a RIPE Atlas API key with the
//! *measurement creation* permission. The key is resolved in this order:
//!
//! 1. `--api_key <KEY>` on the command line;
//! 2. the `MIGREEDY_ATLAS_KEY` environment variable;
//! 3. the key file written by `--save_api_key`.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::PathBuf;

/// Environment variable consulted when no key is given on the command line.
pub const KEY_ENV_VAR: &str = "MIGREEDY_ATLAS_KEY";

/// Directory holding MiGreedy's local configuration.
fn config_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .map(|p| p.join("migreedy"))
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|p| p.join("migreedy"))
    }
}

/// Path of the file holding the RIPE Atlas API key.
pub fn key_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("atlas.key"))
}

/// Read the stored API key, if there is one.
fn stored_key() -> Option<String> {
    let path = key_path()?;
    let contents = fs::read_to_string(path).ok()?;
    let key = contents.trim().to_string();
    (!key.is_empty()).then_some(key)
}

/// Persist the API key so later runs can use it without `--api_key`.
pub fn save_key(key: &str) -> Result<PathBuf> {
    let key = key.trim();
    if key.is_empty() {
        bail!("Refusing to store an empty RIPE Atlas API key.");
    }

    let path = key_path().context("Could not determine a configuration directory to store the API key in (set HOME, or pass --api_key on every run).")?;
    let dir = path.parent().expect("key path always has a parent");
    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create configuration directory {}", dir.display()))?;

    fs::write(&path, format!("{key}\n"))
        .with_context(|| format!("failed to write API key to {}", path.display()))?;

    // The key is a credential: keep it readable by its owner only.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to restrict permissions on {}", path.display()))?;
    }

    Ok(path)
}

/// Resolve the API key from the command line, the environment, or the key file.
pub fn resolve_key(cli_key: Option<&str>) -> Result<String> {
    let from_cli = cli_key.map(str::trim).filter(|k| !k.is_empty());
    if let Some(key) = from_cli {
        return Ok(key.to_string());
    }

    if let Some(key) = std::env::var(KEY_ENV_VAR)
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
    {
        return Ok(key);
    }

    if let Some(key) = stored_key() {
        return Ok(key);
    }

    let stored_at = key_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "the configuration directory".to_string());

    bail!(
        "No RIPE Atlas API key configured.\n\
         Create one with the \"measurement creation\" permission at https://atlas.ripe.net/keys/ and then either:\n  \
         - pass it once with --api_key <KEY> --save_api_key (stored in {stored_at}),\n  \
         - export {KEY_ENV_VAR}=<KEY>,\n  \
         - or pass --api_key <KEY> on every run."
    )
}
