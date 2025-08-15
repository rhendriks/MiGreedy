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

    import math

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
        return "%s\t%s\t%s\t%s\n" % (self._hostname, self._latitude,  self._longitude, self._radius)

class Discs(object):

    def __init__(self):
        self._setDisc={}
        self._orderDisc=collections.OrderedDict()

    def getDiscs(self):
        return self._setDisc

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
