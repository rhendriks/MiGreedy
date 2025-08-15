#!/usr/bin/env python

from disc import *
import collections
import pandas as pd

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

    def find_best_airport(self, airports_df):
        """
        For a given disc, find the best matching airport inside the disc radius.
        It calculates a score based on the population and distance from the disc center.
        Args:
            airports_df: Dataframe containing all airports within the disc radius.
        Returns:
            A list containing the chosen airport's IATA code, latitude, longitude, city, and country code.
            If no suitable airport is found, returns False.
        """
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
