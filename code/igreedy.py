#!/usr/bin/env python
import csv
import os.path
import sys
from pathlib import Path
import numpy as np

from anycast import Anycast
from disc import *

from functools import partial
from math import radians, cos, sin, asin, sqrt

from multiprocessing import Pool
import pandas as pd

import argparse

# Radius of Earth in kilometers (mean radius)
EARTH_RADIUS = 6371.0
FIBER_RI = 1.52
SPEED_OF_LIGHT = 299792.458 # km/s

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
        help='Alpha (population vs distance score tuning) [default: 1.0]'
    )

    parser.add_argument(
        '-t', '--threshold', type=float, default=-1,
        help='Discard disks with RTT > threshold (to bound error) [default: ∞]'
    )

    return parser.parse_args()

def get_airports(path=""):
    if path == "":
        path = os.path.join(os.path.dirname(__file__), '../datasets/airports.csv')

    column_names = [
        'iata', 'size', 'name', 'lat_lon', 'country_code',
        'city', 'pop_heuristic_lon_lat'
    ]
    airports_df = pd.read_csv(
        path,
        sep='\t',
        comment='#',
        names=column_names
    )

    # clean columns
    airports_df[['latitude', 'longitude']] = airports_df['lat_lon'].str.split(expand=True)
    airports_df[['population', 'heuristic', 'google_lon', 'google_lat']] = airports_df['pop_heuristic_lon_lat'].str.split(expand=True)

    # remove unnecessary columns
    airports_df.drop(columns=['lat_lon', 'pop_heuristic_lon_lat', 'size', 'name', 'heuristic', 'google_lon', 'google_lat'], inplace=True)

    # data types
    convert_dict = {
        'latitude': float,
        'longitude': float,
        'population': int
    }
    airports_df = airports_df.astype(convert_dict)

    # index by IATA code
    airports_df.set_index('iata', inplace=True)

    # convert latitude and longitude to radians for geolocation calculations
    airports_df['latitude'] = np.radians(airports_df['latitude'])
    airports_df['longitude'] = np.radians(airports_df['longitude'])

    return airports_df

airports_df = get_airports()  # Load airports data

def analyze(in_df, alpha):
    """
    Routine to iteratively enumerate and geolocate anycast instances
    """
    anycast = Anycast(in_df, airports_df, alpha)

    radiusGeolocated = 0.1
    iteration = True
    discsSolution = []

    numberOfInstance = 0
    while iteration is True:

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

def main(split_dfs, outfile, num_workers, alpha):
    """
    Main routine to run analysis in parallel and write a single aggregated output.
    """
    num_targets = len(split_dfs)
    processed_targets = 0
    all_results = {}

    print(f"Starting parallel processing for {num_targets} targets...")

    with Pool(num_workers) as pool:
        # Use imap_unordered to process results as they are completed
        partial_process = partial(process_target, alpha=alpha)

        for result in pool.imap_unordered(partial_process, split_dfs.items()):
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

def process_target(target_and_df, alpha):
    """
    Worker function for the multiprocessing pool. It analyzes a single target
    and returns its results for aggregation.
    """
    target, split_df = target_and_df
    discsSolution, numberOfInstance = analyze(split_df, alpha)

    # Only return results if they are anycast (more than 1 instance)
    if numberOfInstance > 1:
        return target, discsSolution
    return None

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
                    target, tempCircle.getHostname(), np.degrees(tempCircle.getLatitude()),
                    np.degrees(tempCircle.getLongitude()), tempCircle.getRadius(),
                    tempMarker[0], np.degrees(tempMarker[1]), np.degrees(tempMarker[2])
                ])

if __name__ == "__main__":
    args = parse_args()

    columns = ['target', 'hostname', 'lat', 'lon', 'rtt']
    column_types = {
        'target': str,
        'hostname': str,
        'lat': np.float64,
        'lon': np.float64,
        'rtt': np.float32
    }

    in_df = pd.read_csv(
        args.input,
        skiprows=1,
        names=columns,
        dtype=column_types
    )

    print(f"Input file '{args.input}' loaded. Total records: {len(in_df)}")
    if in_df.empty:
        print("ERROR: Input file is empty or improperly formatted.")
        sys.exit(1)

    # Apply the RTT threshold filter if a positive threshold is provided.
    if args.threshold > 0:
        in_df = in_df[in_df['rtt'] <= args.threshold]

    # Convert lat,lon to radians for consistency
    in_df['lat'] = in_df['lat'].apply(radians)
    in_df['lon'] = in_df['lon'].apply(radians)

    # Calculate the radius in km based on the RTT
    in_df['radius'] = in_df['rtt'] * 0.001 * SPEED_OF_LIGHT / FIBER_RI / 2  # Convert RTT to km

    # create a dictionary of DataFrames, one for each target
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

    main(split_dfs, args.output, num_workers, args.alpha)
