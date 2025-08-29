#!/usr/bin/env python
import os.path
import sys
from pathlib import Path
import numpy as np

from math import radians

import pandas as pd

import argparse
import collections
import math


# Radius of Earth in kilometers (mean radius)
EARTH_RADIUS = 6371.0
FIBER_RI = 1.52
SPEED_OF_LIGHT = 299792.458 # km/s


### Disc class
class Disc(object):
    def __init__(self, hostname, latitude, longitude, radius):
        self._radius = radius
        self._hostname = hostname
        self._latitude = latitude
        self._longitude = longitude

    def getHostname(self):
        return self._hostname

    def getLatitude(self):
        return self._latitude

    def getLongitude(self):
        return self._longitude

    def getRadius(self):
        return self._radius

    def overlap(self, other):
        """
        Two discs overlap if the distance between their centers is lower than
        the sum of their radius.
        """

        return (self.haversine_distance(other._latitude, other._longitude)) <= (self.getRadius() + other.getRadius())

    def haversine_distance(self, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between this disc and another point.

        Args:
            lat2 (float): Latitude of the other point in radians.
            lon2 (float): Longitude of the other point in radians.

        Returns:
            float: Distance between points in kilometers.
        """

        lat1 = self._latitude
        lon1 = self._longitude

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        cos_lat1 = math.cos(lat1)
        cos_lat2 = math.cos(lat2)

        a = (math.sin(dlat / 2) ** 2 +
             cos_lat1 * cos_lat2 * math.sin(dlon / 2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = EARTH_RADIUS * c

        return distance

    def __str__(self):
        return "%s\t%s\t%s\t%s\n" % (self._hostname, self._latitude, self._longitude, self._radius)


class Discs(object):

    def __init__(self):
        self._setDisc = {}
        self._orderDisc = collections.OrderedDict()

    def removeDisc(self, disc):
        self._setDisc[disc[0].getRadius()].remove(disc)

    def overlap(self, other) -> bool:
        """
        Check if any disc in the set overlaps with a given disc.

        Args:
            other: Another disc object to check overlap against.

        Returns:
            bool: True if there is any overlap, False otherwise.
        """
        return any(disc[0].overlap(other) for discs in self._setDisc.values() for disc in discs)

    def add(self, disc, geolocated):
        if self._setDisc.get(disc.getRadius()) is None:
            self._setDisc[disc.getRadius()] = [(disc, geolocated)]
        else:
            self._setDisc[disc.getRadius()].append((disc, geolocated))

    def getOrderedDisc(self):
        self._orderDisc = collections.OrderedDict(sorted(self._setDisc.items()))
        return self._orderDisc
###

### Anycast class
class Anycast(object):
    def __init__(self, in_df, airports, alpha):
        """
        Initializes the object by processing an input DataFrame of network measurements.
        """
        self.alpha = float(alpha)
        self._airports = airports
        self._discsMis = Discs() # disc belong maximum independent set

        # Create a disc for each row in the DataFrame.
        grouped_discs = in_df.groupby('rtt').apply(
            lambda g: [Disc(row.hostname, row.lat, row.lon, row.radius) for row in g.itertuples()],
            include_groups=False
        )

        # A list of discs for each ping, sorted by RTT (lowest first).
        self._orderDisc = collections.OrderedDict(grouped_discs.to_dict())

    def enumeration(self):
        """
        Counts the minimum sets of discs to cover all pings without overlap.
        The number of sets is the number of anycast sites.

        Returns:
            tuple: (number_of_discs_added, _discsMis instance)
        """
        number_of_discs = 0
        discs_mis = self._discsMis

        for ping, discs_set in self._orderDisc.items():
            for disc in discs_set:
                if not discs_mis.overlap(disc):
                    number_of_discs += 1
                    discs_mis.add(disc, False)

        return number_of_discs, discs_mis

    def geolocation(self, disc):
        """
        For a given disc, find the best matching airport inside the disc radius.
        First, it finds all airports within the disc radius.
        Next, it finds the best matching airport based on the population and distance from the disc center.

        Args:
            disc: The disc object.

        Returns:
            The output of geolocateCircle with airports inside the disc.
        """
        radius = disc.getRadius()
        # calculate the haversine distance from the disc center to each airport
        self._airports['dist_from_disc_center'] = self._airports.apply(
            lambda row: disc.haversine_distance(row['latitude'], row['longitude']),
            axis=1
        )
        # filter on airports within the radius
        airports_inside_disk = self._airports[self._airports['dist_from_disc_center'] <= radius].copy()

        if airports_inside_disk.empty:
            return False # no airports within the disc radius

        # calculate population score
        total_pop = airports_inside_disk['population'].sum()
        airports_inside_disk['pop_score'] = airports_inside_disk['population'] / total_pop

        # calculate distance score
        total_distance = airports_inside_disk['dist_from_disc_center'].sum()
        airports_inside_disk['dist_score'] = airports_inside_disk['dist_from_disc_center'] / total_distance
        airports_inside_disk['score'] = self.alpha * airports_inside_disk['pop_score'] + (1 - self.alpha) * airports_inside_disk['dist_score']

        # get chosen airport with the highest score
        best_airport_iata = airports_inside_disk['score'].idxmax()
        chosen_airport = airports_inside_disk.loc[best_airport_iata]

        return [
            best_airport_iata,
            chosen_airport['latitude'],
            chosen_airport['longitude'],
            chosen_airport['city'],
            chosen_airport['country_code']
        ]
###

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

def analyze(in_df, alpha, airports_df):
    """
    Routine to iteratively enumerate and geolocate anycast instances.
    Returns DateFrame with results
    """
    anycast = Anycast(in_df, airports_df, alpha)

    radiusGeolocated = 0.1
    iteration = True
    rows = []  # collect results as dicts

    enumeration_count = 0
    while iteration is True:

        iteration = False
        resultEnumeration = anycast.enumeration()

        enumeration_count += resultEnumeration[0]
        if enumeration_count <= 1:  # unicast
            return None

        for radius, discList in resultEnumeration[1].getOrderedDisc().items():  # anycast
            for disc in discList:
                if not disc[1]:  # if the disc was not geolocated before, geolocate it!
                    resultEnumeration[1].removeDisc(disc)  # remove old disc from MIS
                    city = anycast.geolocation(disc[0])  # result geolocation

                    if city is not False:  # if there is a city inside the disc
                        iteration = True  # geolocated one disc, re-run enumeration!
                        rows.append({
                            "target": in_df['target'].iloc[0],
                            "vp": disc[0].getHostname(),
                            "vp_lat": np.degrees(disc[0].getLatitude()),
                            "vp_lon": np.degrees(disc[0].getLongitude()),
                            "radius": disc[0].getRadius(),
                            "pop_iata": city[0],
                            "pop_lat": np.degrees(city[1]),
                            "pop_lon": np.degrees(city[2]),
                            "pop_city": city[3],
                            "pop_cc": city[4]
                        })
                        resultEnumeration[1].add(
                            Disc("Geolocated", float(city[1]), float(city[2]), radiusGeolocated),
                            True
                        )  # insert the new disc in the MIS
                        break  # exit for rerun MIS
                    else:
                        resultEnumeration[1].add(disc[0], True)  # insert the old disc in the MIS
                        rows.append({
                            "target": in_df['target'].iloc[0],
                            "vp": disc[0].getHostname(),
                            "vp_lat": np.degrees(disc[0].getLatitude()),
                            "vp_lon": np.degrees(disc[0].getLongitude()),
                            "radius": disc[0].getRadius(),
                            "pop_iata": "NoCity",
                            "pop_lat": np.degrees(disc[0].getLatitude()),
                            "pop_lon": np.degrees(disc[0].getLongitude()),
                            "pop_city": "N/A",
                            "pop_cc": "N/A"
                        })

            if iteration:
                break

    return pd.DataFrame(rows)


def main(split_dfs, outfile, alpha):
    """
    Main function to process multiple targets and save results to a file.
    """
    num_targets = len(split_dfs)
    processed_targets = 0
    df_list = []

    airports_df = get_airports()  # Load airports data

    print(f"Starting parallel processing for {num_targets} targets...")

    # perform analysis for each target
    for split_df in split_dfs:
        # run the iGreedy algorithm on the split DataFrame
        discsSolution = analyze(split_df, alpha, airports_df)

        # Only return results if they are anycast (more than 1 instance)
        if discsSolution is not None:
            df_list.append(discsSolution)

        processed_targets += 1
        print(f"Progress: {processed_targets}/{num_targets}", end="\r")

    # After the loop, concatenate all DataFrames
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(outfile, index=False, sep='\t')
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

    print(in_df.head())
    if in_df.empty:
        print("ERROR: Input file is empty or improperly formatted.")
        sys.exit(1)

    # temporary RTT filter for testing
    in_df = in_df[in_df['rtt'] < 20]

    print(f"Records after temporary RTT filter (<20ms): {len(in_df)}")

    # Apply the RTT threshold filter if a positive threshold is provided.
    if args.threshold > 0:
        in_df = in_df[in_df['rtt'] <= args.threshold]

    # Convert lat,lon to radians for consistency
    in_df['lat'] = in_df['lat'].apply(radians)
    in_df['lon'] = in_df['lon'].apply(radians)

    # Calculate the radius in km based on the RTT
    in_df['radius'] = in_df['rtt'] * 0.001 * SPEED_OF_LIGHT / FIBER_RI / 2  # Convert RTT to km TODO save as int?

    # create a dictionary of DataFrames, one for each target
    split_dfs = [group for _, group in in_df.groupby("target")]

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

    main(split_dfs, args.output, args.alpha)
