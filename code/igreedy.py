#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# main user program to process  (./igreedy for help)
# ---------------------------------------------------------------------.
# NOTE: this is a stripped version of the iGreedy script.
# It allows for running the iGreedy script on a directory
# where each file contains latency data towards a target.
# Through multi-threading, removing redundancy, and re-using certain computations,
# this version is able to parse latency data towards the entire v4 space (/24 granularity)
# in several minutes, which before took several hours. -Remi

asciiart = """
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

"""

import os.path

import sys
from anycast import Anycast, Object
from disc import *

from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd


script_dir = os.path.dirname(os.path.realpath(__file__))
iatafile = os.path.join(script_dir, '../datasets/airports.csv')

# ------------------load airport---------------
airports ={}
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


def output(discsSolution, outfile, numberOfInstance):
    """
    Routine to output results to a JSON (for GoogleMaps) and a CSV (for further processing)
    """
    # Results as a CSV
    csv = open(outfile + ".csv", "w")
    csv.write("#hostname\tcircleLatitude\tcircleLongitude\t" + \
              "radius\tiataCode\tiataLatitude\tiataLongitude\n")
    for instance in discsSolution:  # circle to csv
        csv.write(instance[0].getHostname() + "\t" + \
                  str(instance[0].getLatitude()) + "\t" + \
                  str(instance[0].getLongitude()) + "\t" + \
                  str(instance[0].getRadius()) + "\t" + \
                  str(instance[1][0]) + "\t" + \
                  str(instance[1][1]) + "\t" + \
                  str(instance[1][2]) + "\n")
    csv.close()

    # Results as a JSON
    data = Object()
    data.count = numberOfInstance
    data.instances = []
    for instance in discsSolution:
        # circle to Json
        tempCircle = instance[0]
        circle = Object()
        circle.id = tempCircle.getHostname()
        circle.latitude = tempCircle.getLatitude()
        circle.longitude = tempCircle.getLongitude()
        circle.radius = tempCircle.getRadius()
        # marker to Json
        tempMarker = instance[1]
        marker = Object()
        marker.id = tempMarker[0]
        marker.latitude = tempMarker[1]
        marker.longitude = tempMarker[2]
        marker.city = tempMarker[3]
        marker.code_country = tempMarker[4]
        markCircle = Object()
        markCircle.marker = marker
        markCircle.circle = circle
        data.instances.append(markCircle)

    json = open(outfile + ".json", "w")
    json.write("var data=\n")
    json.write(data.to_JSON())
    json.close()


def help():
    print((asciiart + """
usage     (STRIPPED VERSION)
     igreedy.py -i in_df -o output [-a alpha (1)]  [-n noise (0)]  [-t threshold (\infty)] [-m measurement] 

where:
     -i input file (.csv with target IP, hostname of prober, lat of prober, lon of prober, rtt captured from prober to target IP) 
     -o output directory (.csv,.json)
     -a alpha (tune population vs distance score, see INFOCOM'15)
     -n noise (average of exponentially distributed additive latency noise; only for sensitivity)
     -t threshold (discard disks having latency larger than threshold to bound the error)
     """))

    sys.exit()


def main(split_df, outfile):
    alpha = 1
    threshold = -1

    discsSolution, numberOfInstance = analyze(split_df, alpha, threshold)
    if numberOfInstance > 1:  # anycast (unicast results are discarded)
        output(discsSolution, outfile, numberOfInstance)


# Call igreedy script on a single dataframe
def process_file(split_df, out_dir):
    target, df = split_df
    main(df, out_dir +  "/" + target)


if __name__ == "__main__":
    in_file = sys.argv[2]
    out_dir = sys.argv[4]

    in_df = pd.read_csv(in_file, skiprows=1, names=['target', 'hostname', 'lat', 'lon', 'rtt'])

    # split by target (i.e., create a small dataframe for each target)
    split_dfs = {key: group for key, group in in_df.groupby('target')}

    # test on first N targets
    # first_10_targets = list(split_dfs.items())[:10]  # Get the first 10 items from the dictionary
    # split_dfs = dict(first_10_targets)

    # # Make output_directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_targets = in_df['target'].nunique()
    print("Processed files (running iGreedy on this many targets): ", num_targets)

    num_processes = cpu_count()
    print("Number of cpu cores: ", num_processes)

    num_workers = num_processes * 10  # adjust this to increase speed
    processed_targets = 0

    print("Number of workers that will be generated: ", num_processes)
    # Use multiprocessing to run iGreedy in parallel
    partial_process_file = partial(process_file, out_dir=out_dir)
    with Pool(num_workers) as pool:

        for _ in pool.imap_unordered(partial_process_file, list(split_dfs.items())):
            processed_targets += 1

            if processed_targets % 1000 == 0:  # Process bar
                print(f"Progress: {processed_targets:,}/{num_targets:,}", end="\r")
