#!/usr/bin/env python
import os.path
import sys
from pathlib import Path
import numpy as np

from math import radians

import pandas as pd

import argparse
import math


from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

EARTH_RADIUS_KM = 6371.0 # earth radius
FIBER_RI = 1.52
SPEED_OF_LIGHT = 299792.458 # km/s


import math

def haversine(lat1_rad, lon1_rad, other_df):
    """
    Calculates the great-circle distance from a single point to a DataFrame of other points.
    TODO replace math with numpy for speed, once the numpy ufunc issue is resolved.

    Args:
        lat1_rad (float or np.float32): Latitude of the single point in radians.
        lon1_rad (float or np.float32): Longitude of the single point in radians.
        other_df (pd.DataFrame): A DataFrame containing 'lat_rad' and 'lon_rad' columns.

    Returns:
        pd.Series: A Series of distances in kilometers.
    """
    def calculate_single_distance(row):
        # Force all inputs into native Python floats for absolute type safety.
        lat1 = float(lat1_rad)
        lon1 = float(lon1_rad)
        lat2 = float(row['lat_rad'])
        lon2 = float(row['lon_rad'])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = EARTH_RADIUS_KM * c
        return distance

    if other_df.empty:
        return pd.Series(dtype=np.float64) # Handle empty dataframe case

    return other_df.apply(calculate_single_distance, axis=1)

class AnycastDF(object):
    def __init__(self, in_df, airports, alpha):
        """
        Initializes the object using input DataFrames.

        Args:
            in_df (pd.DataFrame): DataFrame with probe measurements including ['hostname', 'lat_rad', 'lon_rad', 'rtt', 'radius'].
            airports (pd.DataFrame): DateFrame with airport data including ['lat_rad', 'lon_rad', 'pop'].
            alpha (float): The weighting factor for the geolocation score.
        """
        self.alpha = float(alpha)

        self._airports = airports

        # Prepare the discs DataFrame, sorted by RTT as required by the algorithm
        self._all_discs_df = in_df.sort_values('rtt').reset_index(drop=True)

        # This DataFrame will store the discs belonging to the maximum independent set (MIS)
        self._mis_df = pd.DataFrame()

    def enumeration(self):
        """
        Finds a maximum independent set of non-overlapping discs using a greedy algorithm.
        The number of discs in the set is the number of anycast sites.

        Returns:
            tuple: (number_of_sites, mis_df) where mis_df is the DataFrame of chosen discs.
        """
        # Start with an empty MIS DataFrame with the same columns and types as _all_discs_df.
        mis_df = pd.DataFrame(columns=self._all_discs_df.columns).astype(self._all_discs_df.dtypes)

        # Iterate through each candidate disc in _all_discs_df.
        for index, candidate_disc in self._all_discs_df.iterrows():
            is_overlapping = False

            # Check for overlap against the MIS DataFrame directly, if it's not empty.
            if not mis_df.empty:
                distances = haversine(
                    candidate_disc['lat_rad'],
                    candidate_disc['lon_rad'],
                    mis_df
                )
                sum_of_radii = candidate_disc['radius'] + mis_df['radius']

                if (distances <= sum_of_radii).any():
                    is_overlapping = True

            if not is_overlapping:
                mis_df.loc[index] = candidate_disc

        self._mis_df = mis_df
        return len(self._mis_df), self._mis_df

    def geolocation(self, disc_series):
        """
        For a given disc (as a Pandas Series), find the best matching airport inside its radius.

        Args:
            disc_series (pd.Series): A row from a DataFrame representing a single disc.
                                     Must include 'lat_rad', 'lon_rad', and 'radius'.

        Returns:
            list or bool: A list with the best airport's details, or False if no airport is found.
        """
        radius = disc_series['radius']

        # Calculate haversine distance from the disc center to ALL airports in a vectorized way
        self._airports['dist_from_disc_center'] = haversine(
            disc_series['lat_rad'],
            disc_series['lon_rad'],
            self._airports
        )

        # Filter on airports within the radius
        airports_inside_disk = self._airports[self._airports['dist_from_disc_center'] <= radius].copy()

        if airports_inside_disk.empty:
            return False  # No airports within the disc radius

        # Calculate scores
        total_pop = airports_inside_disk['pop'].sum()
        if total_pop > 0:
            airports_inside_disk['pop_score'] = airports_inside_disk['pop'] / total_pop
        else:
            airports_inside_disk['pop_score'] = 0

        total_distance = airports_inside_disk['dist_from_disc_center'].sum()

        if total_distance > 0:
            airports_inside_disk['dist_score'] = airports_inside_disk['dist_from_disc_center'] / total_distance
        else:
            airports_inside_disk['dist_score'] = 0

        airports_inside_disk['score'] = self.alpha * airports_inside_disk['pop_score'] - (1 - self.alpha) * \
                                        airports_inside_disk['dist_score']

        # Get chosen airport with the highest score
        best_airport_row = airports_inside_disk.loc[airports_inside_disk['score'].idxmax()]

        return [
            best_airport_row.name, # IATA code
            best_airport_row['lat'],
            best_airport_row['lon'],
            best_airport_row['city'],
            best_airport_row['country_code']
        ]

def parse_args():
    parser = argparse.ArgumentParser(
        description="iGreedy: Geolocation with population-distance tradeoff",
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
    airports_df[['lat', 'lon']] = airports_df['lat_lon'].str.split(expand=True)
    airports_df[['pop', 'heuristic', 'google_lon', 'google_lat']] = airports_df['pop_heuristic_lon_lat'].str.split(expand=True)

    # remove unnecessary columns
    airports_df.drop(columns=['lat_lon', 'pop_heuristic_lon_lat', 'size', 'name', 'heuristic', 'google_lon', 'google_lat'], inplace=True)

    # data types
    convert_dict = {
        'lat': np.float32,
        'lon': np.float32,
        'pop': int
    }
    airports_df = airports_df.astype(convert_dict)

    # index by IATA code
    airports_df.set_index('iata', inplace=True)

    # convert latitude and longitude to radians for geolocation calculations
    airports_df['lat_rad'] = np.radians(airports_df['lat'])
    airports_df['lon_rad'] = np.radians(airports_df['lon'])

    return airports_df


def analyze_df(in_df, alpha, airports_df):
    """
    Routine to iteratively enumerate and geolocate anycast instances using a
    DataFrame-based approach. This version corrects for multiple VPs mapping to the same airport.

    Args:
        in_df (pd.DataFrame): The input measurements.
        alpha (float): The weighting factor for geolocation scoring.
        airports_df (pd.DataFrame): DataFrame of airport data.

    Returns:
        pd.DataFrame: A DataFrame containing the geolocated results, or None if
                      the input is considered unicast.
    """
    anycast = AnycastDF(in_df, airports_df, alpha)
    anycast._all_discs_df['processed'] = False

    radius_geolocated = np.float32(0.1)
    results_rows = []

    # Avoid duplicate airports being geolocated
    chosen_airports = set()

    should_iterate = True
    while should_iterate:
        should_iterate = False

        num_sites, mis_df = anycast.enumeration()

        if num_sites <= 1:
            return None

        # Iterate through the discs in the current MIS, using original index to access master df
        for index, disc_row in mis_df.iterrows():

            # Use the original index of the disc to check its 'processed' status
            original_index = disc_row.name
            if not anycast._all_discs_df.loc[original_index, 'processed']:

                geolocation_result = anycast.geolocation(disc_row)

                if geolocation_result:
                    iata = geolocation_result[0]

                    if iata in chosen_airports:
                        # This airport was already identified by a lower-RTT disc.
                        anycast._all_discs_df.loc[original_index, 'processed'] = True
                        continue

                    chosen_airports.add(iata)
                    should_iterate = True

                    _, lat, lon, city, cc = geolocation_result

                    results_rows.append({
                        "target": in_df['target'].iloc[0],
                        "vp": disc_row['hostname'],
                        "vp_lat": disc_row['lat'],
                        "vp_lon": disc_row['lon'],
                        "radius": disc_row['radius'],
                        "pop_iata": iata,
                        "pop_lat": lat,
                        "pop_lon": lon,
                        "pop_city": city,
                        "pop_cc": cc
                    })

                    master_df = anycast._all_discs_df
                    master_df.loc[original_index, 'lat_rad'] = np.radians(lat)
                    master_df.loc[original_index, 'lon_rad'] = np.radians(lon)
                    master_df.loc[original_index, 'radius'] = radius_geolocated
                    master_df.loc[original_index, 'processed'] = True

                    break  # Break inner 'for' loop to re-run enumeration

                else:  # Geolocation failed (NoCity)
                    anycast._all_discs_df.loc[original_index, 'processed'] = True

                    results_rows.append({
                        "target": in_df['target'].iloc[0],
                        "vp": disc_row['hostname'],
                        "vp_lat": disc_row['lat'],
                        "vp_lon": disc_row['lon'],
                        "radius": disc_row['radius'],
                        "pop_iata": "NoCity",
                        "pop_lat": disc_row['lat'],
                        "pop_lon": disc_row['lon'],
                        "pop_city": "N/A",
                        "pop_cc": "N/A"
                    })

        if should_iterate:
            continue

    if not results_rows:
        return None

    return pd.DataFrame(results_rows)

def process_group(group_tuple, alpha, airports_df):
    """
    Wrapper to be used with pool.imap_unordered.
    It unpacks the (name, group_df) tuple yielded by df.groupby().
    """
    target_name, group_df = group_tuple
    return analyze_df(group_df, alpha, airports_df)

def main(in_df, outfile, alpha):
    """
    Main function to process all targets in parallel.

    Args:
        in_df (pd.DataFrame): The complete input DataFrame containing all targets.
        outfile (str): Path to the output CSV file.
        alpha (float): The alpha parameter for the analysis.
    """
    airports_df = get_airports()  # Load airports data

    num_targets = in_df['target'].nunique()
    print(f"Starting parallel processing for {num_targets} targets using available CPU cores...")    # create a partial function with fixed alpha and airports_df
    worker_func = partial(process_group, alpha=alpha, airports_df=airports_df)

    final_results = []

    # create a pool for the worker processes
    with Pool() as pool:
        # distribute the work and collect results with a progress bar
        results_iterator = pool.imap_unordered(worker_func, in_df.groupby('target'))

        # iterate through results with tqdm progress bar
        for result_df in tqdm(results_iterator, total=num_targets):
            if result_df is not None:
                final_results.append(result_df)

    # After the loop, concatenate all DataFrames
    if final_results:
        print(f"\nFound anycast results for {len(final_results)} targets. Concatenating and saving...")
        final_df = pd.concat(final_results, ignore_index=True)
        final_df.to_csv(outfile, index=False, sep='\t')
        print(f"Results successfully saved to '{outfile}'.")
    else:
        print("\nNo valid anycast results were found to output.")


if __name__ == "__main__":
    args = parse_args()

    columns = ['target', 'hostname', 'lat', 'lon', 'rtt']
    column_types = {
        'target': str,
        'hostname': str,
        'lat': np.float32,
        'lon': np.float32,
        'rtt': np.float32,
    }

    in_df = pd.read_csv(
        args.input,
        skiprows=1, # skip header
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


    print("Adding calculated fields...")
    # Get lat/lon in radians for haversine calculations
    in_df['lat_rad'] = in_df['lat'].apply(radians)
    in_df['lon_rad'] = in_df['lon'].apply(radians)
    # Calculate the radius in km based on the RTT
    in_df['radius'] = in_df['rtt'] * 0.001 * SPEED_OF_LIGHT / FIBER_RI / 2  # Convert RTT to km

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

    main(in_df, args.output, args.alpha)
