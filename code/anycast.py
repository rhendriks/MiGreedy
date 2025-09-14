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

    def geolocation(self, disc_series):
        """
        For a given disc (as a Pandas Series), find the best matching airport inside its radius.
        This version is optimized with a geospatial pre-filtering step.

        Args:
            disc_series (pd.Series): A row from a DataFrame representing a single disc.
                                     Must include 'lat_rad', 'lon_rad', and 'radius'.

        Returns:
            list or bool: A list with the best airport's details, or False if no airport is found.
        """
        radius = disc_series['radius']
        center_lat_rad = disc_series['lat_rad']
        center_lon_rad = disc_series['lon_rad']

        # create bounding box around the disc center
        delta_lat_rad = radius / EARTH_RADIUS_KM
        min_lat_rad = center_lat_rad - delta_lat_rad
        max_lat_rad = center_lat_rad + delta_lat_rad
        delta_lon_rad = radius / (EARTH_RADIUS_KM * np.cos(center_lat_rad))

        min_lon_rad = center_lon_rad - delta_lon_rad
        max_lon_rad = center_lon_rad + delta_lon_rad

        # filter on airports within the bounding box
        candidate_airports = self._airports[
            (self._airports['lat_rad'] >= min_lat_rad) &
            (self._airports['lat_rad'] <= max_lat_rad) &
            (self._airports['lon_rad'] >= min_lon_rad) &
            (self._airports['lon_rad'] <= max_lon_rad)
            ].copy()

        if candidate_airports.empty:
            return False  # No airports even in the rough vicinity.

        # calculate distance from disc center to each candidate airport
        candidate_airports['dist_from_disc_center'] = haversine_numba(
            center_lat_rad,
            center_lon_rad,
            candidate_airports['lat_rad'].to_numpy(dtype=np.float32),
            candidate_airports['lon_rad'].to_numpy(dtype=np.float32),
        )

        # get airports within the disc radius
        airports_inside_disk = candidate_airports[candidate_airports['dist_from_disc_center'] <= radius].copy()

        if airports_inside_disk.empty:
            return False  # No airports within the disc radius

        # calculate population and distance scores
        total_pop = np.float32(airports_inside_disk['pop'].sum())
        if total_pop > 0:
            airports_inside_disk['pop_score'] = airports_inside_disk['pop'] / total_pop
        else:
            airports_inside_disk['pop_score'] = 0

        total_distance = np.float32(airports_inside_disk['dist_from_disc_center'].sum())

        if total_distance > 0:
            airports_inside_disk['dist_score'] = airports_inside_disk['dist_from_disc_center'] / total_distance
        else:
            airports_inside_disk['dist_score'] = 0

        airports_inside_disk['score'] = self.alpha * airports_inside_disk['pop_score'] - (1 - self.alpha) * \
                                        airports_inside_disk['dist_score']

        # select the airport with the highest score
        best_airport_row = airports_inside_disk.loc[airports_inside_disk['score'].idxmax()]

        return [
            best_airport_row.name,  # IATA code
            best_airport_row['lat'],
            best_airport_row['lon'],
            best_airport_row['city'],
            best_airport_row['country_code']
        ]
