#! /usr/bin/python3

## 0: PM2.5 (9), 1: VOC (5), 2: NOx (7), 3: SOx (10), 4: NH3 (15)
## var EmisNames = []string{"VOC", "NOx", "NH3", "SOx", "PM2_5"}

from shapely.geometry import Point, mapping, Polygon
from fiona import collection
import numpy as np

xorig = -2412000
yorig = -1620000
x = np.load('x0.npy')
y = np.load('y0.npy')
x += xorig
y += yorig
emi = np.load('total_emiss.npy')			# Emission results in tonne/yr
emi *= 1.10231							# Conversion to short ton/yr as required by InMAP 

schema = { 'geometry': 'Polygon', 'properties': { 'PM2_5': 'float', 'VOC': 'float','NOx': 'float', 'SOx': 'float', 'NH3': 'float' } }
with collection(
    "testEmis.shp", "w", "ESRI Shapefile", schema) as output:
    for i in range(len(x)-1):
    	for j in range(len(y)-1):
	        bound = [(x[i], y[j]), (x[i], y[j+1]), (x[i+1], y[j+1]), (x[i+1], y[j])]
	        polygon = Polygon(bound)
	        output.write({
	            'properties': {
	                'PM2_5': emi[0][j][i],
	                'VOC' : emi[1][j][i],
	                'NOx' : emi[2][j][i],
	                'SOx' : emi[3][j][i],
	                'NH3' : emi[4][j][i]
	            },
	            'geometry': mapping(polygon)
	        })
