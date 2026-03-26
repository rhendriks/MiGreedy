# A fast, parallel improved version of the iGreedy algorithm for large-scale anycast-aware geolocation
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

[This repository](https://github.com/rhendriks/MiGreedy)
contains a geolocation algorithm based on [iGreedy](https://github.com/fp7mplane/demo-infra/tree/master/igreedy)
that was published in the paper [Latency-Based Anycast Geolocation: Algorithms, Software, and Data Sets](https://ieeexplore.ieee.org/document/7470242).

The delta of this work is a performance aware implementation through multi-threading, implemented in Python and Rust (the latter for performance).
In addition, we improve the iGreedy algorithm by geolocating IPs using the intersection of MIS sets (see iGreedy paper for details) rather than the lowest circle in each set.
This implementation outputs a nearby airport in each MIS set, unicast targets have a single MIS set whereas anycast targets have multiple MIS sets (thus producing multiple airports).

The goal of this implementation is to reduce processing time for [LACeS](https://arxiv.org/abs/2503.20554) (an Open, Fast, Responsible and Efficient Longitudinal Anycast Census System).
This code is used to produce daily anycast censuses, [publicly available](https://github.com/ut-dacs/anycast-census).
It is designed to run using a single input file (containing latencies from multiple vantage points to targets),
outputting a single file with geolocation results.

---

## Pre-compiled binaries

We provide pre-compiled binaries for Linux and macOS.

### Linux (x86_64, static musl)
```bash
curl -LO https://github.com/rhendriks/MiGreedy/releases/latest/download/migreedy-linux-x86_64.tar.gz
tar -xzvf migreedy-linux-x86_64.tar.gz
./migreedy --input path/to/measurements.csv --output path/to/results.csv
```

### macOS (Apple Silicon)
```bash
curl -LO https://github.com/rhendriks/MiGreedy/releases/latest/download/migreedy-macos-aarch64.tar.gz
tar -xzvf migreedy-macos-aarch64.tar.gz
./migreedy --input path/to/measurements.csv --output path/to/results.csv
```

### macOS (Intel)
```bash
curl -LO https://github.com/rhendriks/MiGreedy/releases/latest/download/migreedy-macos-x86_64.tar.gz
tar -xzvf migreedy-macos-x86_64.tar.gz
./migreedy --input path/to/measurements.csv --output path/to/results.csv
```

## Running with Docker

The code can be ran using Docker.

### Step 1: Pull the Docker Image

Pull the latest pre-built image from the GitHub Container Registry:

```bash
docker pull ghcr.io/rhendriks/migreedy:main
```

### Step 2: Prepare Your Data Directory

You need a local directory containing your input CSV file (e.g., `measurements.csv`).
This directory will be mounted into the Docker container.

```bash
# Example: Create a directory and move your data into it
mkdir igreedy_data
mv measurements.csv igreedy_data/
```

### Step 3: Run the Container

Execute the `docker run` command, which mounts your data directory and passes the necessary arguments to the MiGreedy script.

#### Linux/MacOS
```bash
docker run --rm \
  -v "$(pwd)"/igreedy_data:/app/data \
  ghcr.io/rhendriks/migreedy:main \
  --input /app/data/measurements.csv \
  --output /app/data/results.csv
```

#### Windows (PowerShell)
```powershell
docker run --rm `
  -v "${PWD}\igreedy_data:/app/data" `
  ghcr.io/rhendriks/migreedy:main `
  --input /app/data/measurements.csv `
  --output /app/data/results.csv
```

After the command finishes, the output file `results.csv` will appear in your local `igreedy_data` directory.

---

## Installation (Rust)
We include a Rust implementation (significantly faster than the Python version).

1.  Clone this repository:
    ```bash
    git clone [https://github.com/rhendriks/MiGreedy]
    cd [MiGreedy]
    ```
    
2. Install Rust
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    rustup update
    ```

3. Build the project using Cargo:
    ```bash
    cd rust_impl
    cargo build --release
    ```
   
4. Run the compiled binary with the required arguments:
    ```bash
    ./target/release/migreedy --input path/to/measurements.csv --output path/to/results.csv
    ```

## Installation (python)

1.  Clone this repository:
    ```bash
    git clone [https://github.com/rhendriks/MiGreedy]
    cd [MiGreedy]
    ```

2.  Install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script with the required arguments (see below).

    ```bash
    python igreedy.py -i path/to/measurements.csv -o path/to/results.csv -a 1.0 -t 100
    ```

### Command-Line Arguments

| Argument            | Default        | Description                                                                                                                             |
|:--------------------|:---------------|:----------------------------------------------------------------------------------------------------------------------------------------|
| `-i`, `--input`     | **(Required)** | Path to the input CSV file containing RTT measurements. Mutually exclusive with `--atlas`.                                              |
| `--atlas`           |                | RIPE Atlas measurement ID or URL (e.g. `11501` or `https://atlas.ripe.net/measurements/11501/`). Mutually exclusive with `--input`.     |
| `-o`, `--output`    | **(Required)** | Path for the output CSV file where results will be saved. Defaults to `atlas_<ID>.csv` when using `--atlas`.                            |
| `-d`, `--dataset`   | `cities500`    | Location dataset to use: `cities500`, `cities1000`, `cities5000`, `cities15000`, `airports`, or a path to a custom CSV file.            |
| `-a`, `--alpha`     | `1.0`          | A float (0.0 to 1.0) to tune the geolocation scoring. A higher alpha prioritizes population density over distance from the disc center. |
| `-t`, `--threshold` | `0`            | Discards measurements with an RTT greater than this value (in ms) to bound the maximum radius and potential error.                      |
| `--anycast`         | `false`        | If set, outputs only geolocation for anycast targets.                                                                                   |

### RIPE Atlas example

You can geolocate targets directly from a RIPE Atlas measurement without any local data files.
For example, measurement [2001](https://atlas.ripe.net/measurements/2001/) is a traceroute towards K-root:

```bash
./migreedy --atlas 2001
```

This fetches the latest results from the RIPE Atlas API, runs the geolocation algorithm, and writes the output to `atlas_2001.csv`.
You can also pass a full URL instead of a numeric ID:

```bash
./migreedy --atlas https://atlas.ripe.net/measurements/2001/
```

### Datasets

MiGreedy ships with several embedded location datasets. The default (`cities500`) provides the highest geographic coverage, while smaller datasets are faster to process.

| Dataset      | Locations | Min. Population | Description                         |
|:-------------|----------:|----------------:|:------------------------------------|
| `cities500`  |   230,873 |             500 | Maximum coverage (default)          |
| `cities1000` |   167,274 |           1,000 | Good coverage, faster processing    |
| `cities5000` |    68,162 |           5,000 | Balanced coverage and performance   |
| `cities15000`|    33,440 |          15,000 | Fast, suitable for large-scale runs |
| `airports`   |     2,716 |               — | Original airport-only dataset       |

Select a dataset with the `-d` flag:

```bash
./migreedy --atlas 11501 -d cities15000
./migreedy --input measurements.csv --output results.csv -d airports
```

City datasets are sourced from [GeoNames](https://www.geonames.org/) and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Data Format

### Input File Format

The input CSV file **must not have a header** and should contain the following columns in this specific order:

| Column     | Data Type | Description                                |
|:-----------|:----------|:-------------------------------------------|
| `target`   | string    | The IP address being measured.             |
| `hostname` | string    | The hostname or ID of the prober (VP).     |
| `lat`      | float     | The latitude of the prober.                |
| `lon`      | float     | The longitude of the prober.               |
| `rtt`      | float     | The round-trip time (in ms) to the target. |

### Output File Format

The output CSV file will have a header and contain the following columns:

| Column     | Description                                                                                                         |
|:-----------|:--------------------------------------------------------------------------------------------------------------------|
| `target`   | The IP address.                                                                                                     |
| `vp`       | The hostname of the vantage point that defined the disc.                                                            |
| `vp_lat`   | The latitude of the vantage point.                                                                                  |
| `vp_lon`   | The longitude of the vantage point.                                                                                 |
| `radius`   | The radius of the disc in kilometers.                                                                               |
| `pop_iata` | The identifier of the geolocated location (IATA code for airports, GeoNames ID for cities). "NoCity" if none found. |
| `pop_lat`  | The latitude of the geolocated location.                                                                            |
| `pop_lon`  | The longitude of the geolocated location.                                                                           |
| `pop_city` | The city name of the geolocated location.                                                                           |
| `pop_cc`   | The country code of the geolocated location.                                                                        |

---

## Author

*   **Remi Hendriks**
*   **GitHub:** [@rhendriks](https://github.com/rhendriks)
*   **Contact:** `remi.hendriks@utwente.nl`

---

## Contributing
Issues and pull requests are welcome!

## Citation
This code was designed for our paper [LACeS](manycast.net/laces.pdf). Please use the following citation when using this code.
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

