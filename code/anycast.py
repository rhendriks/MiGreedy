#!/usr/bin/env python
#----------------------------------------------------------------------
# detection, enumeration and geolocation helper routines called by the
# main program (igreedy.py)
#---------------------------------------------------------------------.
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


from disc import *
import collections
import json,sys
import random
import pandas as pd

#class for print in Json
class Object:
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


class Anycast(object):
    def __init__(self, in_df, airports, alpha, threshold=-1):
        """
        Initializes the object by processing an input DataFrame of network measurements.

        This method uses a vectorized pandas approach for performance and robustness,
        avoiding slow row-by-row iteration.
        """
        self.alpha = float(alpha)
        self._airports = airports
        self._discsMis = Discs() # disc belong maximum indipendent set

        # --- Vectorized Refactoring Start ---

        # 1. Create a working copy and perform initial data cleaning and type conversion.
        # This is far more robust than using try/except inside a loop.
        df = in_df.copy()
        df['hostname'] = df['hostname'].astype(str) # Ensure hostname is always a string
        
        # Convert numeric columns, coercing errors to NaN (Not a Number)
        numeric_cols = ['lat', 'lon', 'rtt']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows where numeric conversion failed
        df.dropna(subset=numeric_cols, inplace=True)

        # 2. Apply filters to the entire DataFrame at once.
        # Filter out any rows commented with '#'
        df = df[~df['hostname'].str.startswith('#')]

        # Apply the RTT threshold filter if a positive threshold is provided.
        # This is a clearer way to write the original complex condition.
        if threshold > 0:
            df = df[df['rtt'] <= threshold]
        
        # 3. Build the dictionary using a highly efficient `groupby`.
        # This is the modern replacement for the manual loop and dictionary building.
        if not df.empty:
            # Group by 'rtt', then for each group, create a list of Disc objects.
            # .itertuples() is much faster than .iterrows().
            grouped_discs = df.groupby('rtt').apply(
                lambda g: [Disc(row.hostname, row.lat, row.lon, row.rtt) for row in g.itertuples()]
            )
            # The groupby operation sorts the keys (rtt) by default.
            self._setDisc = grouped_discs.to_dict()
        else:
            self._setDisc = {}

        # 4. Create the final ordered dictionary.
        # The keys from the groupby are already sorted, but using OrderedDict is explicit.
        self._orderDisc = collections.OrderedDict(self._setDisc) 
    def detection(self):
        self._discsMis = Discs()
        for ping, setDiscs in self._orderDisc.items(): 
            for disc in setDiscs:
                if not self._discsMis.overlap(disc):
                    self._discsMis.add(disc,False)
                    if(len(self._discsMis)>1):
                        return True
        return False
    
    def enumeration(self):
        numberOfDisc=0
        for ping, setDiscs in self._orderDisc.items(): 
            for disc in setDiscs:
                if not self._discsMis.overlap(disc):
                    numberOfDisc+=1
                    self._discsMis.add(disc,False)
        return [numberOfDisc,self._discsMis]    

    def geolocateCircle(self,disc,airportsSet):
        #alpha parameter for the new igreedy with population
        totalPopulation=0
        totalDistanceFromCenter=0
        chosenCity=""
        oldscore=0
        score=0

        for iata, airportInfo in airportsSet.items(): #_airports[iata]=[float(latitude),float(longitude),int(pop),city,country_code]
            totalPopulation+= airportInfo[2]
            totalDistanceFromCenter+=disc.distanceFromTheCenter(airportInfo[0],airportInfo[1])

        for iata, airportInfo in airportsSet.items(): #_airports[iata]=[float(latitude),float(longitude),int(pop),city,country_code]
            popscore = float(airportInfo[2])/float(totalPopulation)
            distscore = float(disc.distanceFromTheCenter(airportInfo[0],airportInfo[1]))/float(totalDistanceFromCenter)

            #alpha=tunable knob
            score=  self.alpha*popscore + (1-self.alpha)*distscore

            if(score>oldscore):
                chosenCity=[iata,airportInfo[0],airportInfo[1],airportInfo[3],airportInfo[4]]
                oldscore=score
        if(score==0):
            return False
        else:
            return chosenCity
    
    def geolocation(self,disc): 
        airportsInsideDisk={}
        """
        listIataInside=[]
        listPopulation=[] 
        listDistanceFromCenter=[]
        listCityInside=[]
        """

        for iata, airportInfo in self._airports.items(): #_airports[iata]=[float(latitude),float(longitude),int(pop),city,country_code]
            distanceFromBorder=disc.getRadius()-disc.distanceFromTheCenter(airportInfo[0],airportInfo[1])
#create a subset of airport inside the disk and after decide witch one is the one we guess
            if(distanceFromBorder>0): #if the airport is inside the disc
                airportsInsideDisk[iata]=airportInfo

        return self.geolocateCircle(disc,airportsInsideDisk)
        """
                 listIataInside.append(iata)
                 if(airportInfo[3]  not in listCityInside):
                     listPopulation.append(airportInfo[2])
                     listCityInside.append(airportInfo[3])
                     listDistanceFromCenter.append(disc.distanceFromTheCenter(airportInfo[0],airportInfo[1]))
        """

        """
                 if(airportInfo[2]>maxPopulation): #check if the city is more populated
                    geolocatedInstance=[iata,airportInfo[0],airportInfo[1],airportInfo[3],airportInfo[4]]#save the city with the highest population
                    maxPopulation=airportInfo[2] #update the maxPopulation
            elif(maxPopulation==0 and distanceFromBorder>-treshold ): #if there is no city inside yet and the distance is smaller than the threshold 
                if(airportInfo[2]>maxPopulationOut):#check if the city is more populated
                    geolocatedInstanceOut=[iata,airportInfo[0],airportInfo[1]]#save the city with the highest population
                    maxPopulationOut=airportInfo[2] #update the maxPopulation
        """
#-----geolocation fig jsac
#        print disc.getHostname()+"\t"+str(disc.getRadius())+"\t"+str(len(listCityInside))+"\t"+",".join(listIataInside)+"\t"+','.join(str(x) for x in listCityInside)+"\t"+','.join(str(x) for x in listPopulation)+"\t"+','.join(str(x) for x in listDistanceFromCenter)
#-----geolocation fig jsac
        """
        if(maxPopulation!=0):
            return geolocatedInstance
        elif(maxPopulationOut!=0):
            return geolocatedInstanceOut
        return False #no airports inside
        """

