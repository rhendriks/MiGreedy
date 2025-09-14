import pandas as pd
import numpy as np
import os
from numba import jit
from collections import defaultdict

EARTH_RADIUS_KM = 6371.0

@jit(nopython=True)
def haversine_numba(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points in kilometers.
    Expects latitudes and longitudes in radians.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c

def get_airports(path=""):
    if path == "":
        path = os.path.join(os.path.dirname(__file__), '../datasets/airports.csv')

    airports_df = pd.read_csv(
        path,
        sep='\t',
        comment='#',
    )

    print(airports_df.head())

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

def find_nearby_airports(path="./datasets/airports.csv", threshold=100):
    """
    Find airports that are within a certain distance threshold of each other.
    :param threshold: distance threshold in kilometers
    """
    airports = get_airports(path)

    latitudes = airports['lat_rad'].to_numpy()
    longitudes = airports['lon_rad'].to_numpy()
    iata_codes = airports.index.to_numpy()

    n = len(airports)

    # build adjacency list for graph representation
    adjacency_list = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine_numba(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

            if distance <= threshold:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    # find airport clusters using graph traversal (DFS)
    visited = set()
    found_clusters = 0
    for i in range(n):
        if i not in visited:
            cluster_indices = []
            stack = [i]  # Use a stack for iterative DFS

            while stack:
                node_idx = stack.pop()
                if node_idx not in visited:
                    visited.add(node_idx)
                    cluster_indices.append(node_idx)
                    # Add all connected neighbors to the stack to visit them
                    for neighbor_idx in adjacency_list[node_idx]:
                        stack.append(neighbor_idx)

            # print clusters with more than one airport
            if len(cluster_indices) > 1:
                found_clusters += 1

                # get a dataframe for the cluster
                cluster_iata_codes = [iata_codes[k] for k in cluster_indices]
                cluster_df = airports.loc[cluster_iata_codes].copy()
                cluster_df.drop(columns=['lat_rad', 'lon_rad'], inplace=True)
                cluster_df.sort_index(inplace=True)

                print("--- Cluster Found ---")
                print(cluster_df)
                print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    find_nearby_airports(threshold=100) # RTT measurements are accurate up to milliseconds, which translates to ~100 km in fiber optics