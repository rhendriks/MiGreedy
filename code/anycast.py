import numpy as np
import pandas as pd
from numba import jit

EARTH_RADIUS_KM = np.float32(6371.0)  # earth radius

@jit(nopython=True, cache=True)
def haversine_numba(lat1, lon1, lat2_array, lon2_array):
    """
    Calculates the great-circle distance from a single point to an array of other points.
    This function is JIT-compiled by Numba for C-like speed.

    Args:
        lat1 (float): Latitude of the single point in radians.
        lon1 (float): Longitude of the single point in radians.
        lat2_array (np.ndarray): A NumPy array of other latitudes in radians.
        lon2_array (np.ndarray): A NumPy array of other longitudes in radians.

    Returns:
        np.ndarray: A NumPy array of distances in kilometers.
    """

    # Pre-allocate the result array
    distances = np.empty(lat2_array.shape[0], dtype=np.float32)

    # Numba requires explicit loops
    for i in range(lat2_array.shape[0]):
        dlat = lat2_array[i] - lat1
        dlon = lon2_array[i] - lon1

        a = np.sin(dlat / np.float32(2.0)) ** 2 + np.cos(lat1) * np.cos(lat2_array[i]) * np.sin(dlon / np.float32(2.0)) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.float32(1.0) - a))
        distances[i] = EARTH_RADIUS_KM * c

    return distances


class AnycastDF(object):
    def __init__(self, in_df, airports, alpha):
        """
        Initializes the object using input DataFrames.

        Args:
            in_df (pd.DataFrame): DataFrame with probe measurements including ['hostname', 'lat_rad', 'lon_rad', 'rtt', 'radius'].
            airports (pd.DataFrame): DateFrame with airport data including ['lat_rad', 'lon_rad', 'pop'].
            alpha (float): The weighting factor for the geolocation score.
        """
        self.alpha = np.float32(alpha)

        self._airports = airports

        # Sort the input DataFrame by RTT to prioritize lower RTT discs
        self._all_discs_df = in_df.sort_values('rtt').reset_index(drop=True)

        # This DataFrame will store the discs belonging to the maximum independent set (MIS)
        self._mis_df = pd.DataFrame()

    def enumeration(self):
        """
        Finds a maximum independent set of non-overlapping discs using a greedy algorithm.
        The number of discs in the set is the number of anycast sites.

        Assumes that the input DataFrame (_all_discs_df) is sorted by increasing RTT.
        I.e., it prioritizes discs with lower RTT.

        Returns:
            tuple: (number_of_sites, mis_df) where mis_df is the DataFrame of chosen discs.
        """
        # Start with an empty MIS list to put the non-overlapping discs into.
        mis_series_list = []

        # temporary lists for calculating overlaps
        mis_lats = []
        mis_lons = []
        mis_radii = []

        # Iterate through each candidate disc in _all_discs_df.
        for index, candidate_disc in self._all_discs_df.iterrows():
            is_overlapping = False

            # check for overlap with existing discs in the MIS
            if mis_series_list: # only check if MIS is not empty
                # create arrays from the MIS lists
                lat2_array = np.array(mis_lats, dtype=np.float32)
                lon2_array = np.array(mis_lons, dtype=np.float32)

                # get distance to all existing discs in the MIS
                distances = haversine_numba(
                    candidate_disc['lat_rad'],
                    candidate_disc['lon_rad'],
                    lat2_array,
                    lon2_array
                )

                # calculate sum of radii (i.e., overlap threshold)
                radii_array = np.array(mis_radii, dtype=np.float32)
                sum_of_radii = candidate_disc['radius'] + radii_array

                if np.any(distances <= sum_of_radii):
                    is_overlapping = True

            if not is_overlapping:
                # add to MIS if no overlap with any existing disc
                mis_lats.append(candidate_disc['lat_rad'])
                mis_lons.append(candidate_disc['lon_rad'])
                mis_radii.append(candidate_disc['radius'])
                mis_series_list.append(candidate_disc)

        # store the MIS as a DataFrame
        if mis_series_list:
            self._mis_df = pd.DataFrame(mis_series_list, index=[s.name for s in mis_series_list])
        else:
            self._mis_df = pd.DataFrame(columns=self._all_discs_df.columns).astype(self._all_discs_df.dtypes)
        return len(self._mis_df), self._mis_df

    def build_cluster(self, mis_disc_series):
        """
        Returns all unprocessed discs from _all_discs_df that overlap with the given MIS disc
        (i.e., distance between centres ≤ sum of their radii).

        These discs all likely measure the same anycast site, so their intersection
        provides a tighter geolocation constraint than the MIS disc alone.

        Unprocessed-only filter: already-located site proxies (radius shrunk to 0.1 km)
        are excluded so they do not over-constrain clusters in later iterations.

        Args:
            mis_disc_series (pd.Series): A row from _mis_df representing the MIS disc.

        Returns:
            pd.DataFrame: The subset of unprocessed discs in _all_discs_df that overlap
                          with the given disc. Always contains at least the MIS disc itself.
        """
        unprocessed = self._all_discs_df[~self._all_discs_df['processed']]
        distances = haversine_numba(
            mis_disc_series['lat_rad'],
            mis_disc_series['lon_rad'],
            unprocessed['lat_rad'].to_numpy(dtype=np.float32),
            unprocessed['lon_rad'].to_numpy(dtype=np.float32),
        )
        overlap_mask = distances <= (mis_disc_series['radius'] + unprocessed['radius'].to_numpy(dtype=np.float32))
        return unprocessed[overlap_mask]

    def geolocation(self, cluster_df):
        """
        For a cluster of discs representing the same anycast site, find the best matching
        airport within their intersection (i.e., within the radius of every disc in the cluster).

        The smallest disc anchors the bounding-box pre-filter and the distance scoring,
        as it provides the tightest single constraint.

        Args:
            cluster_df (pd.DataFrame): Rows from _all_discs_df representing one site's cluster.
                                       Must include 'lat_rad', 'lon_rad', and 'radius'.

        Returns:
            list or bool: [iata, lat, lon, city, country_code] or False if no airport is found.
        """
        # Smallest disc = tightest single constraint; used for bounding box and distance scoring
        smallest_idx = cluster_df['radius'].idxmin()
        smallest_disc = cluster_df.loc[smallest_idx]

        radius = smallest_disc['radius']
        center_lat_rad = smallest_disc['lat_rad']
        center_lon_rad = smallest_disc['lon_rad']

        # Bounding box pre-filter based on the smallest disc
        delta_lat_rad = radius / EARTH_RADIUS_KM
        min_lat_rad = center_lat_rad - delta_lat_rad
        max_lat_rad = center_lat_rad + delta_lat_rad
        delta_lon_rad = radius / (EARTH_RADIUS_KM * np.cos(center_lat_rad))
        min_lon_rad = center_lon_rad - delta_lon_rad
        max_lon_rad = center_lon_rad + delta_lon_rad

        candidate_airports = self._airports[
            (self._airports['lat_rad'] >= min_lat_rad) &
            (self._airports['lat_rad'] <= max_lat_rad) &
            (self._airports['lon_rad'] >= min_lon_rad) &
            (self._airports['lon_rad'] <= max_lon_rad)
        ].copy()

        if candidate_airports.empty:
            return False  # No airports even in the rough vicinity.

        # Intersect: retain only airports within every disc in the cluster
        for _, disc in cluster_df.iterrows():
            dists = haversine_numba(
                disc['lat_rad'], disc['lon_rad'],
                candidate_airports['lat_rad'].to_numpy(dtype=np.float32),
                candidate_airports['lon_rad'].to_numpy(dtype=np.float32),
            )
            candidate_airports = candidate_airports[dists <= disc['radius']]
            if candidate_airports.empty:
                return False  # Intersection is empty.

        # Score airports; distance is measured from the smallest disc's centre
        candidate_airports['dist_from_disc_center'] = haversine_numba(
            center_lat_rad, center_lon_rad,
            candidate_airports['lat_rad'].to_numpy(dtype=np.float32),
            candidate_airports['lon_rad'].to_numpy(dtype=np.float32),
        )

        total_pop = np.float32(candidate_airports['pop'].sum())
        if total_pop > 0:
            candidate_airports['pop_score'] = candidate_airports['pop'] / total_pop
        else:
            candidate_airports['pop_score'] = np.float32(0)

        total_distance = np.float32(candidate_airports['dist_from_disc_center'].sum())
        if total_distance > 0:
            candidate_airports['dist_score'] = candidate_airports['dist_from_disc_center'] / total_distance
        else:
            candidate_airports['dist_score'] = np.float32(0)

        candidate_airports['score'] = (
            self.alpha * candidate_airports['pop_score']
            - (1 - self.alpha) * candidate_airports['dist_score']
        )

        # select the airport with the highest score
        best_airport_row = candidate_airports.loc[candidate_airports['score'].idxmax()]

        return [
            best_airport_row.name,  # IATA code
            best_airport_row['lat'],
            best_airport_row['lon'],
            best_airport_row['city'],
            best_airport_row['country_code']
        ]
