# Changelog

All notable changes to MiGreedy are documented in this file.

## [Unreleased]

### Changed
- Updated dependencies (polars, clap, rayon, indicatif, reqwest, serde).
- Clarified the RIPE Atlas measurement example in the README.

## [1.5.3] - 2026-03-28

### Added
- **RIPE Atlas DNS measurement support** (`--atlas`) — DNS measurement results are
  now parsed in addition to ping measurements.

## [1.5.2] - 2026-03-28

Reworks geolocation around the cities dataset and makes the candidate search
substantially faster.

### Added
- **Relative population threshold** (`-p`/`--pop-ratio`) — during geolocation only
  cities with `pop >= max_pop × ratio` are kept among the candidates. Combines with
  the absolute `--min-pop` threshold, which filters at load time.
- **Accuracy output** (`--accuracy`) — adds `candidate_diameter` (km) and
  `num_constraints` columns to the output.

### Changed
- **Single cities file** — the per-population-threshold city datasets were replaced
  by one dataset (all cities with population ≥ 500) plus runtime filtering.
- **Eligible locations are found with an R-tree** (`rstar`) instead of a linear scan.
- **Computations are parallelized** with rayon, and repeated MIS distance
  calculations, disc string allocations and candidate-vector clones were removed.
- **Coordinates are stored in radians**, keeping only the disc radius.
- **Discs overlapping multiple MIS sets are excluded from clusters**, since it is
  unknown which anycast site they reach.
- The Rust code was split into modules (`analyzer`, `atlas`, `geo`, `io`, `model`)
  and documented.

### Fixed
- **RIPE Atlas API fetching**.
- **Dataset copying** in the Docker build.

### Removed
- **Python implementation** — the tool is Rust-only.
- **`clean_airports` script** — superseded by the shipped datasets.

## [1.5.0] - 2026-03-26

Adds direct RIPE Atlas input and city-based geolocation.

### Added
- **RIPE Atlas measurement input** (`--atlas`) — pass a measurement ID or URL
  (e.g. `2001` or `https://atlas.ripe.net/measurements/2001/`) instead of `--input`.
  Results are fetched from the RIPE Atlas API and written to `atlas_<ID>.csv`.
- **Cities dataset** (`-d cities`, default) — geolocation iteratively searches for
  the best city instead of only mapping to airports.
- **Embedded compressed datasets** — the airports and cities files are gzip-compressed
  and embedded in the binary, so no external data files are needed.
- **macOS release targets** (`aarch64-apple-darwin`, `x86_64-apple-darwin`).
- **Testing code**.

### Fixed
- **Dockerfile** dataset handling.
- **Incorrect citation** in the README.

## [1.4.3] - 2026-02-27

### Added
- **Anycast-only output** (`--anycast`) — only geolocations for anycast targets are written.
- **Embedded airports file** — the airports dataset is included in the binary.

### Changed
- Updated polars.

## [1.4.0] - 2026-02-27

### Added
- **Unicast geolocation** — targets detected as unicast are geolocated and written out.
- **MIS intersection geolocation** — geolocation uses the intersection of the
  maximum independent set of discs.

### Changed
- Release workflow simplified (checksum steps removed, artifact packaging reworked).
- Dockerfile reworked for the Rust build; stripping is done by cargo.
- README download link and `tar` command corrected.

## [1.2.3] - 2026-01-16

### Changed
- Version bump.

## [1.2.2] - 2026-01-16

### Changed
- Release workflow artifact naming.

## [1.2.1] - 2026-01-16

### Added
- **`CITATION.cff`** with the LACeS citation.
- **Release profile settings** in `Cargo.toml` (strip, opt-level, LTO, panic=abort);
  release binaries are tagged with the version.

### Fixed
- **Country codes for Rwanda and Kosovo** (contributed by @m-appel).

## [1.2.0] - 2025-09-14

### Changed
- **Grouping is parallelized** and moved out of input-data reading.

## [1.1.0] - 2025-09-14

### Changed
- Release workflow now also pushes a `latest` tag.

## [1.0.3] - 2025-09-14

### Fixed
- **Build and packaging paths** in the release workflow.

## [1.0.1] - 2025-09-14

### Added
- **Release workflow** producing a static MUSL binary.

## [1.0.0] - 2025-09-14

- Initial Rust release, replacing the Python implementation.
- Docker image (amd64/arm64) built from a stripped static MUSL binary.
- Airports dataset cleaned up: duplicates and airports within 100 km of each other removed (keeping the largest).