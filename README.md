# MiGreedy

```text
180 150W  120W  90W   60W   30W  000   30E   60E   90E   120E  150E 180
|    |     |     |     |     |    |     |     |     |     |     |     |
+90N-+-----+-----+-----+-----+----+-----+-----+-----+-----+-----+-----+
|          . _..::__:  ,-"-"._       |7       ,     _,.__             |
|  _.___ _ _<_>`!(._`.`-.    /        _._     `_ ,_/  '  '-._.---.-.__|
|.{     " " `-==,',._\{  \  / {)     / _ ">_,-' `                mt-2_|
+ \_.:--.       `._ )`^-. "'      , [_/( G        e      o     __,/-' +
|'"'     \         "    _L       0o_,--'                )     /. (|   |
|         | A  n     y,'          >_.\\._<> 6              _,' /  '   |
|         `. c   s   /          [~/_'` `"(   l     o      <'}  )      |
+30N       \\  a .-.t)          /   `-'"..' `:._        c  _)  '      +
|   `        \  (  `(          /         `:\  > \  ,-^.  /' '         |
|             `._,   ""        |           \`'   \|   ?_)  {\         |
|                `=.---.       `._._ i     ,'     "`  |' ,- '.        |
+000               |a    `-._       |     /          `:`<_|h--._      +
|                  (      l >       .     | ,          `=.__.`-'\     |
|                   `.     /        |     |{|              ,-.,\     .|
|                    |   ,'          \ z / `'            ," a   \     |
+30S                 |  /             |_'                |  __ t/     +
|                    |o|                                 '-'  `-'  i\.|
|                    |/                                        "  n / |
|                    \.          _                              _     |
+60S                            / \   _ __  _   _  ___ __ _ ___| |_   +
|                     ,/       / _ \ | '_ \| | | |/ __/ _` / __| __|  |
|    ,-----"-..?----_/ )      / ___ \| | | | |_| | (_| (_| \__ \ |_ _ |
|.._(                  `----'/_/   \_\_| |_|\__, |\___\__,_|___/\__| -|
+90S-+-----+-----+-----+-----+-----+-----+--___/ /--+-----+-----+-----+
     Based on 1998 Map by Matthew Thomas   |____/ Hacked on 2015 by 8^/
```

MiGreedy is an anycast-aware IP geolocation implementation using latency measurements.
Originally designed as a multi-threaded (hence the 'M') and optimized implementation of the
[iGreedy](https://github.com/fp7mplane/demo-infra/tree/master/igreedy) algorithm published in
[Latency-Based Anycast Geolocation: Algorithms, Software, and Data Sets](https://ieeexplore.ieee.org/document/7470242).
In particular, this was created for the [LACeS](https://manycast.net) daily anycast census
as the python iGreedy implementation struggled with the scale and frequency of measurements.

This has been extended by:
* Implemented in Rust for speed and resource efficiency
* Unicast geolocation alongside anycast geolocation
* Improved accuracy of anycast geolocation using intersection of discs in MIS clusters
* Support for geolocation at city granularity in addition to airports
* Confidence and accuracy metrics (`--accuracy`)
* Improved live RIPE Atlas measurement support through VP selection algorithms

**Measurement input**
* CSV (optionally gzipped) and Parquet files, including [MAnycastR](https://github.com/rhendriks/MAnycastR) latency output
* scamper warts files, read natively — used by the LACeS pipeline
* RIPE Atlas measurements, fetched by ID or scheduled live against a target

This README is the manual for installing and running MiGreedy.

## Contents

* [Installation](#installation)
* [Quick start](#quick-start)
* [How it works](#how-it-works)
* [Input formats](#input-formats)
* [RIPE Atlas](#ripe-atlas)
* [Datasets](#datasets)
* [Output format](#output-format)
* [Options reference](#options-reference)
* [Contributing](#contributing)
* [Citation](#citation)

## Installation

### Download a binary

Pre-compiled binaries are available for Linux and macOS.

**Linux (x86_64, static musl)**

```bash
curl -LO https://github.com/rhendriks/MiGreedy/releases/latest/download/migreedy-linux-x86_64.tar.gz
tar -xzvf migreedy-linux-x86_64.tar.gz
```

**macOS (Apple Silicon)**

```bash
curl -LO https://github.com/rhendriks/MiGreedy/releases/latest/download/migreedy-macos-aarch64.tar.gz
tar -xzvf migreedy-macos-aarch64.tar.gz
```

**macOS (Intel)**

```bash
curl -LO https://github.com/rhendriks/MiGreedy/releases/latest/download/migreedy-macos-x86_64.tar.gz
tar -xzvf migreedy-macos-x86_64.tar.gz
```

### Docker

```bash
docker pull ghcr.io/rhendriks/migreedy:main
```

The container reads and writes in a mounted data directory:

```bash
mkdir igreedy_data
mv measurements.csv igreedy_data/

docker run --rm \
  -v "$(pwd)"/igreedy_data:/app/data \
  ghcr.io/rhendriks/migreedy:main \
  --input /app/data/measurements.csv \
  --output /app/data/results.csv
```

On Windows (PowerShell):

```powershell
docker run --rm `
  -v "${PWD}\igreedy_data:/app/data" `
  ghcr.io/rhendriks/migreedy:main `
  --input /app/data/measurements.csv `
  --output /app/data/results.csv
```

The output file appears in the mounted directory once the run finishes.

### Build from source

Requires rustup.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source $HOME/.cargo/env
git clone https://github.com/rhendriks/MiGreedy.git && cd MiGreedy
cargo build --release
```

The binary is written to `target/release/migreedy`.

## Quick start

**1. Geolocate a measurement file** (CSV, gzipped CSV, Parquet, or warts):

```bash
./migreedy --input measurements.csv --output results.csv
```

Each output row is one geolocated site: a unicast target produces a single row, an anycast target
one row per detected site. See [Input formats](#input-formats) and [Output format](#output-format).

**2. Or geolocate an existing RIPE Atlas measurement**, by ID or URL, with no local data files:

```bash
./migreedy --atlas 2001
```

**3. Or schedule your own measurement** towards a target, which needs a RIPE Atlas API key:

```bash
./migreedy --measure 1.1.1.1
```

## How it works

Given a set of RTT (round-trip time) measurements from geographically distributed vantage points
(VPs) to a target IP, the algorithm determines the target's location(s):

1. **RTT to distance** — Each VP's RTT is converted to a disc using the speed of light in fiber. The target must lie somewhere within this disc.

2. **Enumeration (MIS)** — Discs are sorted by radius (ascending). Every non-overlapping MIS disc geolocates an anycast replica (a single MIS disc means unicast).

3. **Clustering** — For each MIS disc, all other discs that overlap with it (and only this MIS disc) are collected into a *cluster*. These discs all likely measured the same site.

4. **Intersection & geolocation** — Within the cluster:
   - Find the **smallest disc** in the cluster (tightest constraint). This is always the MIS disc itself.
   - Collect all candidate cities inside that disc.
   - Progressively intersect with each cluster disc (smallest to largest). If adding the next disc would remove all candidates, stop and use the last non-empty set.
   - Apply the **relative population filter** (`--pop_ratio`): keep only cities with `pop >= max_pop × ratio`, where `max_pop` is the largest population among remaining candidates.
   - Select the best city using: `score = α × (pop / Σpop) − (1 − α) × (dist / Σdist)`, where distance is measured from the disc's center.

5. **Output** — One row per detected site, each with the geolocated city, its coordinates, and the MIS disc VP.

### Accuracy trade-off: intersection vs. single-disc geolocation

The original iGreedy algorithm geolocates each site from its MIS disc alone.
We instead intersect the MIS disc with the other discs in its cluster (step 4 above).
Both approaches are approximations, and they fail in different ways.

**Single-disc (iGreedy).**
The geolocation of an anycast site is located within the MIS disc that found it.
In some cases the MIS disc may be very large, in which case the accuracy of the geolocation is affected.
I.e., there may be many candidate cities, and it will pick the most likely based on population and distance.
Our analysis shows this often leads to false geolocations, especially when ran on latency data collected using VPs with sparse coverage in certain regions.

**Intersection (this implementation).**
Large MIS discs are likely to have intersections with other non-MIS discs.
We use those to intersect our MIS disc and narrow the possible location of the anycast site reached.
This substantially improves geolocation in regions with sparse coverage where multiple VPs reach the same site (with medium to high latencies).
However, this may also lead to false geolocations e.g., when an intersecting non-MIS disc is reaching a different anycast site (creating a fake intersecting area).
We limit the occurrence of this by only using intersecting discs that intersect only this MIS disc, but it may still happen.

**Why we keep the intersection.**
Both methods may lead to false geolocations within the MIS disc.
However, for our large scale census we find it improves accuracy.
Intersecting discs also allows us to output accurate unicast geolocations.

## Input formats

Exactly one input source is required: `--input`, `--atlas`, `--warts`, or `--measure`.

### CSV

The input CSV file **must have a header row**, and its columns are read positionally in this order:

| Column     | Data type | Description                                |
|------------|-----------|--------------------------------------------|
| `addr`     | string    | The IP address being measured.             |
| `hostname` | string    | The hostname or ID of the prober (VP).     |
| `lat`      | float     | The latitude of the prober.                |
| `lon`      | float     | The longitude of the prober.               |
| `rtt`      | float     | The round-trip time (in ms) to the target. |

When `--vps` is given, the `lat` and `lon` columns are looked up from the VPs file
instead and must be omitted, leaving `addr,hostname,rtt`.

A path ending in `.gz` is decompressed first, so a gzipped CSV is read directly:

```bash
./migreedy --input measurements.csv.gz --output results.csv
```

### Parquet

A path ending in `.parquet` is read as Parquet. Parquet files carry their own column
names, so unlike CSV their columns are matched **by name** and in any order.

| Column              | Required | Description                                                        |
|---------------------|----------|--------------------------------------------------------------------|
| `addr`              | yes      | The IP address being measured, as text or as packed address bytes. |
| `hostname`, or `rx` | yes      | The hostname or ID of the prober (VP).                             |
| `rtt`               | yes      | The round-trip time (in ms) to the target.                         |
| `lat`, `lon`        | no       | The prober's coordinates. Without them, `--vps` is required.       |

This reads [MAnycastR](https://github.com/rhendriks/MAnycastR) latency output as it is
written, whose columns are `rx, addr, ttl, rtt`.
MAnycastR stores each address as 16 IPv6-mapped bytes rather than as text.
This format is supported.

### VPs file

A VPs file gives each vantage point's location, so measurements that identify their
VP only by name can be turned into discs. It is required with `--warts` and optional
with `--input`.

The format is whitespace-separated `hostname lat lon`, one per line, with **no header**:

```text
hlz2-nz.ark.caida.org -37.79 175.28
fra-de.ark.caida.org 50.11 8.74
hkg4-cn.ark.caida.org 22.36 114.12
```

### Warts

`--warts` reads [scamper](https://www.caida.org/catalog/software/scamper/) output
directly. `.warts` and `.warts.gz` are both supported.

```bash
# a directory of files
./migreedy --warts /data/2026-08-10/ --vps vps.txt --output results.csv

# explicit files, or a quoted glob
./migreedy --warts a.warts b.warts.gz --vps vps.txt --output results.csv
./migreedy --warts '/data/*.iffinder.warts.gz' --vps vps.txt --output results.csv
```

The vantage point for each file is taken from the monitor name recorded inside the
file, falling back to the filename if that name is not one the VPs file lists.

This only supports `dealias` records.

## RIPE Atlas

### Geolocating an existing measurement

Targets can be geolocated directly from a RIPE Atlas measurement.
For example, measurement [2001](https://atlas.ripe.net/measurements/2001/) is a periodic ping towards K-root:

```bash
./migreedy --atlas 2001
```

This fetches the latest results from the RIPE Atlas API, runs the geolocation algorithm, and writes
the output to `atlas_2001.csv`. A full URL works instead of a numeric ID:

```bash
./migreedy --atlas https://atlas.ripe.net/measurements/2001/
```

> **NOTE:** RIPE Atlas probes may have wrong user-reported locations which result in wrong
> geolocation results. `--atlas` uses every probe in the measurement as-is; `--measure`
> screens them first (see below).

### Scheduling a new measurement

`--measure` takes a target instead of a measurement ID: MiGreedy picks the probes,
schedules a one-off ping, waits for the results and geolocates them in one go.

**Configuring an API key.**
Scheduling measurements needs a RIPE Atlas API key with the **measurement creation**
permission, which you can make at [atlas.ripe.net/keys](https://atlas.ripe.net/keys/).
Store it once:

```bash
./migreedy --api_key <YOUR-KEY> --save_api_key
```

The key is written to `~/.config/migreedy/atlas.key` with owner-only permissions.

**Measuring a target.**

```bash
./migreedy --measure 1.1.1.1
```

This selects 100 probes, pings the target from each of them, and writes the geolocated
sites to `atlas_<ID>.csv`, where `<ID>` is the measurement RIPE Atlas created.

```bash
./migreedy --measure 1.1.1.1 8.8.8.8 9.9.9.9 --num_probes 200 --output results.csv
```

Use `--dry_run` to see which probes would be used without scheduling anything.

```bash
./migreedy --measure 1.1.1.1 --num_probes 20 --dry_run
```

### Choosing probes

We maximize geographical spread when choosing probes to improve geolocation accuracy.
This is done using greedy farthest-point sampling, which is reported.

```text
Selected 20 probes for 1 IPv4 target(s): 1.1.1.1.
Coverage: 19 countries, closest pair 3265 km apart, 17/20 probes on well-connected networks.
```

Alternatively, you can use your own list of probes.
E.g., if you know the target is within Europe, you can create a list of European probes.

```bash
./migreedy --measure 1.1.1.1 --probes 6118,1010358,23002
./migreedy --measure 1.1.1.1 --probes my-probes.txt
```

### Filtering probes with implausible locations

RIPE Atlas probes have self reported geolocations.
These can be inaccurate, which would result in wrong geolocation output.

> **NOTE:** A good method is to verify probe locations with geolocation databases.

### Validating probe locations using anchors

Using `--validate_probes` verifies probe location validity using anchor measurements.
It pings five globally spread anchors from the candidate probe.
Probes reporting a speed-of-light violation to any of the anchors are dropped.

> **NOTE:** Few anchors have false geolocations, which would invalidate this check.

```bash
./migreedy --measure 1.1.1.1 --validate_probes
```

This is off by default as it incurs additional measurements.

Because validation removes probes, MiGreedy selects 10% more candidates than asked for
and keeps up to `--num_probes` validated probes.

## Datasets

MiGreedy ships with embedded airports and cities datasets.
The cities dataset contains all cities with a population of 500 or higher (sourced from GeoNames).

Select a dataset with the `-d` flag:

```bash
./migreedy --atlas 11501 -d cities
./migreedy --atlas 11501 -d cities --min_pop 15000
./migreedy --input measurements.csv --output results.csv -d airports
```

**Population filtering**

* `--min_pop <N>` filters cities globally at load time (absolute threshold)
* `--pop_ratio <R>` filters cities per-geolocation, keeping only those with `pop >= max_pop × R` (relative threshold)

These can be combined. For example, `--min_pop 10000 --pop_ratio 0.5` first removes all cities under 10k,
then during each geolocation keeps only the top 50% by population among candidates.

City datasets are sourced from [GeoNames](https://www.geonames.org/) and licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Output format

Results are written to the path given by `-o` as tab-separated CSV by default,
or as Parquet when the path ends in `.parquet`.
Rows are sorted by address, so repeated runs of the same input produce identical files.

| Column     | Parquet type | Description                                                                                 |
|------------|--------------|---------------------------------------------------------------------------------------------|
| `addr`     | binary       | The IP address.                                                                             |
| `vp`       | string       | The hostname of the vantage point that defined the disc.                                    |
| `vp_lat`   | float32      | The latitude of the vantage point.                                                          |
| `vp_lon`   | float32      | The longitude of the vantage point.                                                         |
| `radius`   | float32      | The radius of the disc in kilometers.                                                       |
| `pop_iata` | string       | The identifier of the geolocated location (IATA code for airports, GeoNames ID for cities). |
| `pop_lat`  | float32      | The latitude of the geolocated location. The vantage point's latitude if none found.        |
| `pop_lon`  | float32      | The longitude of the geolocated location. The vantage point's longitude if none found.      |
| `pop_city` | string       | The city name of the geolocated location.                                                   |
| `pop_cc`   | string       | The country code of the geolocated location.                                                |

With `--accuracy`, two columns are appended:

| Column               | Parquet type | Description                                                                                                  |
|----------------------|--------------|--------------------------------------------------------------------------------------------------------------|
| `candidate_diameter` | float32      | Maximum pairwise distance (km) between surviving candidate cities. Smaller values indicate higher precision. |
| `num_constraints`    | uint16       | Number of discs that narrowed the candidate set. Higher values indicate higher confidence in the result.     |

When no location was found for a site, CSV writes `NoCity`, `N/A` and `0` in the location and
accuracy columns, and Parquet leaves them null.

**Parquet files** `addr` is stored as 16 packed bytes, with IPv4 written IPv6-mapped (`::ffff:1.1.1.1`).

## Options reference

Also available as `migreedy --help`.

**Input** (exactly one source is required)

| Option                   | Default | Description                                                                                        |
|--------------------------|---------|----------------------------------------------------------------------------------------------------|
| `-i`, `--input <PATH>`   |         | Input CSV (optionally `.gz`) or `.parquet` file containing RTT measurements                        |
| `--atlas <ID>`           |         | RIPE Atlas measurement ID or URL (e.g. `11501` or `https://atlas.ripe.net/measurements/11501/`)    |
| `--warts <PATHS>`        |         | scamper warts files (`.warts`/`.warts.gz`): files, glob patterns or directories. Requires `--vps`  |
| `--measure <TARGETS>`    |         | Target(s) to measure live: schedules RIPE Atlas pings and geolocates the results. Needs an API key |
| `--vps <PATH>`           |         | Vantage point coordinates file. Required with `--warts`; rejected with `--atlas` and `--measure`   |
| `-t`, `--threshold <MS>` | `0`     | Discard measurements with an RTT above this value (in ms), bounding the maximum radius and error   |

**Geolocation**

| Option                   | Default  | Description                                                                                  |
|--------------------------|----------|----------------------------------------------------------------------------------------------|
| `-d`, `--dataset <NAME>` | `cities` | Location dataset: `cities` (embedded), `airports` (embedded), or a path to a custom CSV file |
| `-m`, `--min_pop <N>`    | `0`      | Absolute minimum population. Cities below this are filtered out at load time                 |
| `-p`, `--pop_ratio <R>`  | `0.0`    | Relative population threshold (0.0–1.0): keeps candidates with `pop >= max_pop × ratio`      |
| `-a`, `--alpha <A>`      | `1.0`    | Scoring weight (0.0–1.0). Higher prioritizes population over distance from the disc center   |

**Output**

| Option                  | Default        | Description                                                                                                      |
|-------------------------|----------------|------------------------------------------------------------------------------------------------------------------|
| `-o`, `--output <PATH>` | **(Required)** | Output file; `.parquet` is written as Parquet, anything else as CSV. Defaults to `atlas_<ID>.csv` with `--atlas` |
| `--anycast`             | off            | Only output geolocations for anycast targets                                                                     |
| `--accuracy`            | off            | Add the `candidate_diameter` (km) and `num_constraints` columns                                                  |

**RIPE Atlas measurements** (`--measure` only)

| Option                        | Default | Description                                                                                      |
|-------------------------------|---------|--------------------------------------------------------------------------------------------------|
| `--api_key <KEY>`             |         | RIPE Atlas API key with the *measurement creation* permission                                    |
| `--save_api_key`              | off     | Store `--api_key` for later runs                                                                 |
| `--num_probes <N>`            | `100`   | How many probes to select, spread for the widest global coverage                                 |
| `--probes <IDS>`              |         | Measure from these probes instead: comma-separated IDs, or a file listing them                   |
| `--packets <N>`               | `1`     | Ping packets sent per probe                                                                      |
| `--measurement_timeout <SEC>` | `300`   | Seconds to wait for results before continuing with whatever has arrived                          |
| `--validate_probes`           | off     | Also ping anchors to drop probes whose location the measured RTTs rule out. Costs extra credits  |
| `--dry_run`                   | off     | Report the probe selection and exit without scheduling anything (and without needing an API key) |

## Contributing

Issues and pull requests are welcome.

Maintained by Remi Hendriks ([@rhendriks](https://github.com/rhendriks), `remi.hendriks@utwente.nl`).

## Citation

MiGreedy was developed for the [following paper](https://manycast.net/laces.pdf).
Please cite it when using MiGreedy.

```
@inproceedings{10.1145/3730567.3764484,
      author = {Hendriks, Remi and Luckie, Matthew and Jonker, Mattijs and Sommese, Raffaele and van Rijswijk-Deij, Roland},
      title = {LACeS: An Open, Fast, Responsible and Efficient Longitudinal Anycast Census System},
      year = {2025},
      isbn = {9798400718601},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3730567.3764484},
      doi = {10.1145/3730567.3764484},
      abstract = {IP anycast replicates an address at multiple locations to reduce latency and enhance resilience. Due to anycast's crucial role in the modern Internet, earlier research introduced tools to perform anycast censuses. The first, iGreedy, uses latency measurements from geographically dispersed locations to map anycast deployments. The second, MAnycast2, uses anycast to perform a census of other anycast networks. MAnycast2's advantage is speed and coverage but suffers from problems with accuracy, while iGreedy is highly accurate but slower using author-defined probing rates and costlier. In this paper we address the shortcomings of both systems and present LACeS (Longitudinal Anycast Census System). Taking MAnycast2 as a basis, we completely redesign its measurement pipeline, and add support for distributed probing, additional protocols (DNS over UDP, TCP SYN/ACK, and IPv6) and latency measurements similar to iGreedy. We validate LACeS on an anycast testbed with 32 globally distributed nodes, compare against an external anycast production deployment, extensive latency measurements with RIPE Atlas and cross-check over 60\% of detected anycast using operator ground truth that shows LACeS achieves high accuracy. Finally, we provide a longitudinal analysis of anycast, covering 17+months, showing LACeS achieves high precision. We make continual daily LACeS censuses available to the community and release the source code of the tool under a permissive open source license.},
      booktitle = {Proceedings of the 2025 ACM Internet Measurement Conference},
      pages = {445–461},
      numpages = {17},
      keywords = {internet measurement, anycast, internet topology, routing, ip},
      location = {USA},
      series = {IMC '25}
}
```

