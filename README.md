# A fast, parallel implementations of the iGreedy algorithm for large-scale anycast census

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

[This repository](https://github.com/rhendriks/MiGreedy) contains a multiprocessing implementation of the iGreedy anycast geolocation algorithm,
originally developed by [Cicalsese et al.](https://github.com/fp7mplane/demo-infra/tree/master/igreedy)
that was published in the paper [Latency-Based Anycast Geolocation: Algorithms, Software, and Data Sets](https://ieeexplore.ieee.org/document/7470242).

The goal of this implementation is to reduce processing time for [large-scale anycast censuses](github.com/anycast-census/anycast-census).
It is designed to run using a single input file (containing latencies from multiple vantage points to targets),
outputting a single file with geolocation results.

---

## Running with Docker

The code can be ran using Docker.

### Step 1: Pull the Docker Image

Pull the latest pre-built image from the GitHub Container Registry:

```bash
docker pull ghcr.io/rhendriks/igreedy:latest
```

### Step 2: Prepare Your Data Directory

You need a local directory containing your input CSV file (e.g., `measurements.csv`).
This directory will be mounted into the Docker container.

```bash
# Example: Create a directory and move your data into it
mkdir igreedy_data
mv measurements.csv igreedy_data/
```

### Step 3: Run the Script

Execute the `docker run` command, which mounts your data directory into the container's `/app/data` directory. The script then reads from and writes to this folder.

The command below will:
*   Run the container and automatically remove it once it's finished (`--rm`).
*   Mount your current working directory's `igreedy_data` subfolder to `/app/data` inside the container (`-v`).
*   Execute the iGreedy script with the correct paths *inside the container*.

```bash
# For Linux / macOS
docker run --rm \
  -v "$(pwd)"/igreedy_data:/app/data \
  ghcr.io/rhendriks/igreedy:latest \
  -i /app/data/measurements.csv \
  -o /app/data/results.csv \
  -a 0.8
```

```bash
# For Windows (Command Prompt)
docker run --rm ^
  -v "%cd%\igreedy_data":/app/data ^
  ghcr.io/rhendriks/igreedy:latest ^
  -i /app/data/measurements.csv ^
  -o /app/data/results.csv ^
  -a 0.8
```

### Step 4: Results

After the command finishes, the output file (`results.csv`) will appear in your local `igreedy_data` directory.

---

## Installation (Rust)
We include a Rust implementation (10X+ faster than Python version).

1.  Clone this repository:
    ```bash
    ```bash
    git clone [https://github.com/rhendriks/MiGreedy]
    cd [MiGreedy]
    ```
    
2. Install Rust
    '''bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    rustup update
    '''

3. Build the project using Cargo:
    ```bash
    cd rust_impl
    cargo build --release
    ```
   
4. Run the compiled binary with the required arguments:
    ```bash
    ./target/release/migreedy --input path/to/measurements.csv --output path/to/results.csv --airports ../datasets/airports.csv
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

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | **(Required)** | Path to the input CSV file containing RTT measurements. |
| `-o`, `--output` | **(Required)** | Path for the output CSV file where results will be saved. |
| `-a`, `--alpha` | `1.0` | A float (0.0 to 1.0) to tune the geolocation scoring. A higher alpha prioritizes population density over distance from the disc center. |
| `-t`, `--threshold`| `None` | Discards measurements with an RTT greater than this value (in ms) to bound the maximum radius and potential error. |

## Data Format

### Input File Format

The input CSV file **must not have a header** and should contain the following columns in this specific order:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `target` | string | The anycast IP address being measured. |
| `hostname` | string | The hostname or ID of the prober (VP). |
| `lat` | float | The latitude of the prober. |
| `lon` | float | The longitude of the prober. |
| `rtt` | float | The round-trip time (in ms) to the target. |

### Output File Format

The output CSV file will have a header and contain the following columns:

| Column | Description |
| :--- | :--- |
| `target` | The anycast IP address. |
| `vp` | The hostname of the vantage point that defined the disc. |
| `vp_lat` | The latitude of the vantage point. |
| `vp_lon` | The longitude of the vantage point. |
| `radius` | The radius of the disc in kilometers. |
| `pop_iata` | The IATA code of the geolocated airport. "NoCity" if none found. |
| `pop_lat` | The latitude of the geolocated airport. |
| `pop_lon` | The longitude of the geolocated airport. |
| `pop_city` | The city of the geolocated airport. |
| `pop_cc` | The country code of the geolocated airport. |

---

## Author

*   **Remi Hendriks**
*   **GitHub:** [@rhendriks](https://github.com/rhendriks)
*   **Contact:** `remi.hendriks@utwente.nl`

---

## Contributing
Issues and pull requests are welcome!
