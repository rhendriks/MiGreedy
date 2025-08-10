#!/usr/bin/env python

import math,collections
import numpy as np

EARTH_RADIUS = 6371.0


class Disc(object):
    def __init__(self, hostname, latitude, longitude, radius):
        self._radius    = radius
        self._hostname  = hostname
        self._latitude=latitude
        self._longitude=longitude

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
        
        return (self.haversine_distance(other._latitude,other._longitude)) <= (self.getRadius() + other.getRadius())

    def haversine_distance(self, lat2, lon2):
        """
        Calculates the great-circle distance between two points on Earth.

        Args:
            self: The current disc object with latitude and longitude.
            lat2_rad, lon2_rad: Latitude and longitude of other disc.

        Returns:
            float: The distance between the two points in kilometers.
        """

        # lat1_rad = math.radians(self._latitude)
        # lon1_rad = math.radians(self._longitude)
        #
        # lon2_rad = math.radians(lon2)
        # lat2_rad = math.radians(lat2)

        dlon = lon2 - self._longitude
        dlat = lat2 - self._latitude

        a = np.sin(dlat / 2.0) ** 2 + np.cos(self._latitude) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = EARTH_RADIUS * c

        return distance


    def __str__(self):
        return "%s\t%s\t%s\t%s\n" % (self._hostname, self._latitude,  self._longitude, self._radius)

class Discs(object):

    def __init__(self):
        self._setDisc={}
        self._orderDisc=collections.OrderedDict()

    def getDiscs(self):
        return self._setDisc

    def removeDisc(self, disc):
        self._setDisc[disc[0].getRadius()].remove(disc)

    def overlap(self, other):

        for radius, listDisc in self._setDisc.items():
            
            for disc in listDisc:
                if disc[0].overlap(other):
                    return True
        return False
    

    def add(self,disc,geolocated):
        if self._setDisc.get(disc.getRadius()) is None:
           self._setDisc[disc.getRadius()]=[(disc,geolocated)]
        else:
            self._setDisc[disc.getRadius()].append((disc,geolocated))

    def getOrderedDisc(self):
        self._orderDisc=collections.OrderedDict(sorted(self._setDisc.items()))
        return self._orderDisc

    def smallestDisc(self):
        self._orderDisc=collections.OrderedDict(sorted(self._setDisc.items()))
        return next(iter(self._orderDisc))
