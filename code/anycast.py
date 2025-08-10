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

        # --- Vectorized Refactoring Start ---

        # 3. Build the dictionary using a highly efficient `groupby`.
        # This is the modern replacement for the manual loop and dictionary building.
        # Group by 'rtt', then for each group, create a list of Disc objects.
        # .itertuples() is much faster than .iterrows().
        grouped_discs = in_df.groupby('rtt').apply(
            lambda g: [Disc(row.hostname, row.lat, row.lon, row.radius) for row in g.itertuples()],
            include_groups=False
        )
        # The groupby operation sorts the keys (rtt) by default.
        self._setDisc = grouped_discs.to_dict()

        # 4. Create the final ordered dictionary.
        # The keys from the groupby are already sorted, but using OrderedDict is explicit.
        self._orderDisc = collections.OrderedDict(self._setDisc)

    def enumeration(self):
        numberOfDisc=0
        for ping, setDiscs in self._orderDisc.items(): 
            for disc in setDiscs:
                if not self._discsMis.overlap(disc):
                    numberOfDisc+=1
                    self._discsMis.add(disc,False)
        return [numberOfDisc,self._discsMis]    

    def geolocateCircle(self,disc,airportsSet):
        totalPopulation=0
        totalDistanceFromCenter=0
        chosenCity=""
        oldscore=0
        score=0

        for iata, airportInfo in airportsSet.items(): #_airports[iata]=[float(latitude),float(longitude),int(pop),city,country_code]
            totalPopulation+= airportInfo[2]
            totalDistanceFromCenter+=disc.haversine_distance(airportInfo[0],airportInfo[1])

        for iata, airportInfo in airportsSet.items(): #_airports[iata]=[float(latitude),float(longitude),int(pop),city,country_code]
            popscore = float(airportInfo[2])/float(totalPopulation)
            distscore = float(disc.haversine_distance(airportInfo[0],airportInfo[1]))/float(totalDistanceFromCenter)

            #alpha=tunable knob
            score=  self.alpha*popscore + (1-self.alpha)*distscore

            if score > oldscore:
                chosenCity=[iata,airportInfo[0],airportInfo[1],airportInfo[3],airportInfo[4]]
                oldscore=score
        if score==0:
            return False
        else:
            return chosenCity
    
    def geolocation(self,disc): 
        airportsInsideDisk={}

        for iata, airportInfo in self._airports.items(): #_airports[iata]=[float(latitude),float(longitude),int(pop),city,country_code]
            distanceFromBorder=disc.getRadius()-disc.haversine_distance(airportInfo[0],airportInfo[1])
#create a subset of airport inside the disk and after decide witch one is the one we guess
            if(distanceFromBorder>0): #if the airport is inside the disc
                airportsInsideDisk[iata]=airportInfo

        return self.geolocateCircle(disc,airportsInsideDisk)
