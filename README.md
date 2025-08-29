# A fast, parallel implementations of the iGreedy algorithm for large-scale anycast census

<pre>
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
</pre>

This repository contains a multiprocessing implementation of the iGreedy anycast geolocation algorithm,
originally developed by [Cicalsese et al.](https://github.com/fp7mplane/demo-infra/tree/master/igreedy)

The goal of this implementation is to reduce processing time for [large-scale anycast censuses](github.com/anycast-census/anycast-census).
It is designed to run using a single input file (containing latencies from multiple vantage points to targets),
outputting a single file with geolocation results.


## Installation

1.  Clone this repository:
    ```bash
    git clone [your-repository-url]
    cd [repository-folder]
    ```

2.  Install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You should create a `requirements.txt` file by running `pip freeze > requirements.txt` in your development environment).*

## Usage

The script is run from the command line, with arguments to specify the input and output files, as well as tuning parameters.

```bash
python igreedy.py -i path/to/measurements.csv -o path/to/results.csv
```

### TODOs
* update iata airports file
* test and re-implement RIPE Atlas
* create automated testing with sample latencies file
* load in list of VPs from file, rather than having the lat and lon repeated for each data point