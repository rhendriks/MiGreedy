# Changelog

All notable changes to MiGreedy are documented in this file.

## [1.7.0] - 2026-09-01

Adds live RIPE Atlas measurements: give MiGreedy a target and it schedules the ping
itself. Measurement files can now also be read as Parquet.

### Added
- **Gzipped CSV input** (`--input <FILE>.csv.gz`) — gzip now supported.
- **Parquet input** (`--input <FILE>.parquet`) — parquet now supported.
  Including native support for MAnycastR latency measurement files.
- **Live measurements** (`--measure`) — schedules a one-off RIPE Atlas ping to one or
  more targets, waits for the results, and geolocates them in the same run.
- **Probe selection** (`--num_probes`, `--probes`) — probes are chosen by greedy
  farthest-point sampling, maximising the minimum distance between them for coverage.
  Selects stable probes only, and prefers well-connected probes in CDN networks or anchors.
- **Probe location validation** (`--validate_probes`) — optionally pings five globally
  spread anchors and drops probes whose reported location is farther from an anchor than
  the measured RTT allows at the speed of light in fibre. Selects 10% extra candidates
  first so coverage survives the removals.

## [1.6.0] - 2026-08-13

Adds scamper warts input support and allows for VPs files rather than repeated lat/lon values within CSV files.

### Added
- **Warts input** (`--warts`) — accepts files, glob patterns and directories for `.warts` and `.warts.gz` files.
  Files are parsed in parallel. Currently, only Alias-resolution (`dealias`) records are supported.
- **Vantage point coordinates file** (`--vps`) — whitespace-separated
  `hostname lat lon`, no header. Required with `--warts` which contains no `lat/lon` values. 
  Also usable with CSV files (`--input`) so the CSV can omit its `lat`/`lon` columns.

### Changed
- **Filtering and deduplication** - measurements with a missing or non-positive RTT are dropped.
  When there are multiple RTT results per (`addr`, `hostname`) pair, only the minimum RTT result is kept.

### Fixed
- **Deduplication preserves input order**, so the vantage point credited for a site
  no longer varies between runs when two discs share a radius exactly.

## [1.5.4] - 2026-08-10

### Changed
- **Fixed `--accuracy` quadratic cost** - `candidate_diameter` was computed by
  comparing every pair of surviving candidates. This was expensive for large MIS discs.
  We now approximate the distance for large MIS discs using a farthest-point sweep.

### Documentation
- Added a section on the accuracy trade-off between disc intersection and single-disc
  (iGreedy) geolocation.

## [1.5.3] - 2026-03-28

### Added
- **RIPE Atlas DNS measurement support** (`--atlas`) — DNS measurement results are
  now parsed in addition to ping measurements.

## [1.5.2] - 2026-03-28

Reworks geolocation around the cities dataset and makes the candidate search
substantially faster.

### Added
- **Relative population threshold** (`-p`/`--pop_ratio`) — during geolocation only
  cities with `pop >= max_pop × ratio` are kept among the candidates. Combines with
  the absolute `--min_pop` threshold, which filters at load time.
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