#!/usr/bin/env python
import csv
import json
import os.path
import sys
from pathlib import Path

from anycast import Anycast, Object
from disc import *

from multiprocessing import Pool
from functools import partial
import pandas as pd

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="iGreedy: Geolocation with population-distance tradeoff (STRIPPED VERSION)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i', '--input', required=True,
        help='Input CSV file containing:\n'
             '    - target IP\n'
             '    - hostname of prober\n'
             '    - latitude and longitude of prober\n'
             '    - RTT from prober to target IP'
    )

    parser.add_argument(
        '-o', '--output', required=True,
        help='Output file (e.g., results.csv)'
    )

    parser.add_argument(
        '-a', '--alpha', type=float, default=1.0,
        help='Alpha (population vs distance score tuning, see INFOCOM\'15) [default: 1.0]'
    )

    parser.add_argument(
        '-n', '--noise', type=float, default=0.0,
        help='Average of exponentially distributed additive latency noise (for sensitivity) [default: 0.0]'
    )

    parser.add_argument(
        '-t', '--threshold', type=float, default=-1,
        help='Discard disks with RTT > threshold (to bound error) [default: ∞]'
    )

    parser.add_argument(
        '-m', '--measurement', default=None,
        help='Optional measurement label or ID (optional)'
    )

    return parser.parse_args()


script_dir = os.path.dirname(os.path.realpath(__file__))
iatafile = os.path.join(script_dir, '../datasets/airports.csv')

# ------------------load airport---------------
airports = {}
with open(iatafile, 'r', encoding='utf-8') as airportLines:
    airportLines.readline()
    for line in airportLines.readlines():
        iata, size, name, latLon, country_code, city, popHeuristicGooglemapslonlat = line.strip().split("\t")
        latitude, longitude = latLon.strip().split()
        pop, Heuristic, lon, lat = popHeuristicGooglemapslonlat.strip().split()
        airports[iata] = [float(latitude), float(longitude), int(pop), city, country_code]
airportLines.close()


def analyze(in_df, alpha, threshold):
    """
    Routine to iteratively enumerate and geolocate anycast instances
    """
    anycast = Anycast(in_df, airports, alpha, threshold)

    radiusGeolocated = 0.1
    iteration = True
    discsSolution = []

    numberOfInstance = 0
    while (iteration is True):

        iteration = False
        resultEnumeration = anycast.enumeration()

        numberOfInstance += resultEnumeration[0]
        if (numberOfInstance <= 1):
            # print("No anycast instance detected")
            return 0, 1
        for radius, discList in resultEnumeration[1].getOrderedDisc().items():
            for disc in discList:
                if (not disc[1]):  # if the disc was not geolocated before, geolocate it!
                    # used for the csv output
                    # discs.append(disc[0])#append the disc to the results #used for the csv output
                    resultEnumeration[1].removeDisc(disc)  # remove old disc from MIS of disc
                    city = anycast.geolocation(disc[0])  # result geolocation

                    if (city != False):  # if there is a city inside the disc
                        iteration = True  # geolocated one disc, re-run enumeration!
                        # markers.append(newDisc)#save for the results the city#used for the csv output
                        discsSolution.append((disc[0], city))  # disc, marker
                        resultEnumeration[1].add(Disc("Geolocated", float(city[1]), float(city[2]), radiusGeolocated),
                                                 True)  # insert the new disc in the MIS

                        break  # exit for rerun MIS
                    else:
                        resultEnumeration[1].add(disc[0], True)  # insert the old disc in the MIS
                        discsSolution.append((disc[0], ["NoCity", disc[0].getLatitude(), disc[0].getLongitude(), "N/A",
                                                        "N/A"]))  # disc, marker

            if (iteration):
                break
    return discsSolution, numberOfInstance

def main(split_dfs, outfile, num_workers):
    """
    Main routine to run analysis in parallel and write a single aggregated output.
    """
    num_targets = len(split_dfs)
    processed_targets = 0
    all_results = {}

    print(f"Starting parallel processing for {num_targets} targets...")

    with Pool(num_workers) as pool:
        # Use imap_unordered to process results as they are completed
        for result in pool.imap_unordered(process_target, split_dfs.items()):
            # Collect valid results returned by the workers
            if result:
                target, discsSolution = result
                all_results[target] = discsSolution

            processed_targets += 1
            print(f"Progress: {processed_targets}/{num_targets}", end="\r")

    # After the pool is finished, write the single aggregated files
    if all_results:
        output_aggregated(all_results, outfile)
        print(f"\nProcessing complete. Results saved to {outfile}")
    else:
        print("\nNo valid anycast results were found to output.")

def process_target(target_and_df):
    """
    Worker function for the multiprocessing pool. It analyzes a single target
    and returns its results for aggregation.
    """
    alpha = 1
    threshold = -1

    target, split_df = target_and_df
    discsSolution, numberOfInstance = analyze(split_df, alpha, threshold)

    # Only return results if they are anycast (more than 1 instance)
    if numberOfInstance > 1:
        return target, discsSolution
    return None

# Call igreedy script on a single dataframe
def process_file(split_df, out_dir):
    target, df = split_df
    main(df, out_dir +  "/" + target)


def output_aggregated(all_results, outfile):
    """
    Writes aggregated results from all targets to a single JSON and a single CSV file.

    Args:
        all_results (dict): A dictionary where keys are targets and values are their 'discsSolution'.
        outfile_prefix (str): The base name for the output .csv and .json files.
    """
    # 1. Write aggregated results to a single CSV file
    print(f"\nWriting aggregated CSV to {outfile}...")
    with open(outfile, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        # Write header with the new 'target' column
        writer.writerow(["target", "vp", "vp_lat", "vp_lon", "radius",
                         "pop_iata", "pop_lat", "pop_lon"])

        # Iterate through each target and its solution
        for target, discsSolution in all_results.items():
            for instance in discsSolution:
                tempCircle, tempMarker = instance[0], instance[1]
                writer.writerow([
                    target, tempCircle.getHostname(), tempCircle.getLatitude(),
                    tempCircle.getLongitude(), tempCircle.getRadius(),
                    tempMarker[0], tempMarker[1], tempMarker[2]
                ])

if __name__ == "__main__":
    args = parse_args()

    in_df = pd.read_csv(args.input, skiprows=1, names=['target', 'hostname', 'lat', 'lon', 'rtt'])

    in_df['hostname'] = in_df['hostname'].astype(str)  # Ensure hostname is always a string

    # Convert numeric columns, coercing errors to NaN (Not a Number)
    numeric_cols = ['lat', 'lon', 'rtt']
    for col in numeric_cols:
        in_df[col] = pd.to_numeric(in_df[col], errors='coerce')

    # Drop any rows where numeric conversion failed
    in_df.dropna(subset=numeric_cols, inplace=True)

    # Apply the RTT threshold filter if a positive threshold is provided.
    if args.threshold > 0:
        in_df = in_df[in_df['rtt'] <= args.threshold]

    # split by target (i.e., create a small dataframe for each target)
    split_dfs = {key: group for key, group in in_df.groupby('target')}

    output_loc = Path(args.output)
    output_dir = output_loc.parent

    # Check whether the desired output file can be created
    if not output_dir.exists():
        print(f"Output directory '{output_dir}' does not exist. Attempting to create it...")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"ERROR: Permission denied. Cannot create directory '{output_dir}'.")
            sys.exit(1)
        except OSError as e:
            print(f"ERROR: Could not create directory '{output_dir}'. OS error: {e}")
            sys.exit(1)

    # Verify permissions
    if not os.access(output_dir, os.W_OK):
        print(f"ERROR: Permission denied. Cannot write to directory '{output_dir}'.")
        sys.exit(1)

    if output_loc.is_dir():
        print(f"ERROR: Output path '{output_loc}' is a directory, not a file. Please specify a file name.")
        sys.exit(1)

    # Check for overwriting
    if output_loc.exists():
        print(f"Output file '{output_loc}' already exists. Overwriting...")
    else:
        print(f"Output file '{output_loc}' will be created in '{output_dir}'.")

    num_targets = in_df['target'].nunique()
    print("Processed files (running iGreedy on this many targets): ", num_targets)

    num_workers = os.cpu_count()
    print("Number of cpu cores: ", num_workers)

    main(split_dfs, args.output, num_workers)
