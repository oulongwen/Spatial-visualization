import networkx as nx
import json
from shapely.geometry import LineString
from shapely.geometry import shape
from shapely.geometry import Polygon

from mpl_toolkits.basemap import Basemap
from math import floor
import matplotlib.pyplot as plt
import numpy as np
from numpy import meshgrid
from rtree import index

import fiona
import cvxopt
from cvxopt import solvers, matrix, spmatrix, sparse
# import cvxopt.solvers
import cvxopt.glpk

class Emiss:
    def __init__(self, nrows = 82, ncols = 132, df = 1):
        self.lon = []
        self.lat = []
        self.x = []
        self.y = []
        self.prod = []
        self.df = df
        self.emi = np.zeros((5, nrows, ncols))


class Map:
    
    def __init__(self, grid = 12):
        
        # Define domain
        if grid == 12:
            self.ncols = 396
            self.nrows = 246
            self.dx = 12000
            self.dy = 12000
        if grid == 36:
            self.ncols = 132
            self.nrows = 82
            self.dx = 36000
            self.dy = 36000
        else:
            pass
        self.g = nx.read_shp('/Users/oulongwen/Downloads/in15oc03/in101503.shp')
        self.sg = list(nx.connected_component_subgraphs(self.g.to_undirected()))[0]
        
    def GenMap(self,lonmin = -97, latmin = 40, lonmax = -50, latmax = 54, xorig = -2412000, yorig = -1620000):
        self.xorig = xorig
        self.yorig = yorig
        self.lonmin = lonmin
        self.latmin = latmin
        self.lonmax = lonmax
        self.latmax = latmax

        self.mapgen = Basemap(llcrnrlon=self.lonmin, llcrnrlat=self.latmin, urcrnrlon=self.lonmax,
                              urcrnrlat=self.latmax, projection='lcc', rsphere = 6370000, lat_1 = 33, lat_2 = 45, 
                              lat_0=40, lon_0=-97, resolution='l', area_thresh=10000)

        self.lonorig, self.latorig = self.mapgen(xorig,yorig,inverse = True)
        self.lonmax, self.latmax = self.mapgen(xorig+(self.ncols)*self.dx, yorig+(self.nrows)*self.dy, inverse = True)

        self.mapgen = Basemap(llcrnrlon=self.lonorig, llcrnrlat=self.latorig, urcrnrlon=self.lonmax,
                              urcrnrlat=self.latmax, projection='lcc', rsphere = 6370000, lat_1 = 33, lat_2 = 45, 
                              lat_0=40, lon_0=-97, resolution='l', area_thresh=10000)

        # The first parameter of makegird determines the number of columns,
        # while the second parameter determines the number of rows of the grid.
        # These are number of points, not cells.
        self.lons,self.lats,self.x,self.y=self.mapgen.makegrid(self.ncols+1,self.nrows+1,returnxy = True)
        self.mapgen.drawcoastlines()
        self.mapgen.drawcountries(linewidth=2)
        self.mapgen.drawstates()

        return self.mapgen
    
    def GenDic(self):
        self.idx = index.Index()
        self.squares = []
        ids = 0
        self.tup = []
        x0 = self.x[0,:]
        y0 = self.y[:,0]
        for i in range(len(x0)-1):
            for j in range(len(y0)-1):
                bounds = (x0[i], y0[j], x0[i+1], y0[j+1])
                self.idx.insert(ids,bounds)
                temp_tup = (ids,(i,j))
                self.tup.append(temp_tup)
                ids += 1
        self.id_dict = dict(self.tup)
        return self.id_dict
    
    def pipe(self):
        m = self.mapgen
        m.readshapefile('/Users/oulongwen/Downloads/CrudeOil_Pipelines_US_EIA/CrudeOil_Pipelines_US_Nov2014','test')
        x0 = self.x[0,:]
        y0 = self.y[:,0]
        pipetest = np.zeros((self.nrows,self.ncols))
        self.total_oilpipe = 0
        for item in m.test:
            ring = LineString(item)
            self.total_oilpipe += ring.length
            for code in list(set(self.idx.intersection(ring.bounds))):
                i,j = self.id_dict[code]
                square = Polygon([(x0[i],y0[j]),(x0[i+1],y0[j]),(x0[i+1],y0[j+1]),(x0[i],y0[j+1])])
                sect = ring.intersection(square)
                if len(sect.bounds)>0:
                    pipetest[round(square.bounds[1]/self.dy)][round(square.bounds[0]/self.dx)] += sect.length
        self.oilpipe = pipetest
        
        lines = fiona.open('/Users/oulongwen/Downloads/NaturalGas_InterIntrastate_Pipelines_US_EIA/NaturalGas_Pipelines_US_2015.shp')
        string = []
        for line in lines:
            coords = line['geometry']['coordinates']
            try:
                bb=list(zip(*coords))[0:2]
                temp = m(bb[0],bb[1])
                string.append(LineString(list(zip(temp[0],temp[1]))))
            except:
                for i in coords:
                    try:
                        bb=list(zip(*i))[0:2]
                        temp = m(bb[0],bb[1])
                        string.append(LineString(list(zip(temp[0],temp[1]))))
                    except:
                        pass
        pipetest = np.zeros((self.nrows,self.ncols))
        self.total_ngpipe = 0
        for ring in string:
            self.total_ngpipe += ring.length
            for code in list(set(self.idx.intersection(ring.bounds))):    # Remove duplicate values in idx.intersection results
                i,j = self.id_dict[code]
                square = Polygon([(x0[i],y0[j]),(x0[i+1],y0[j]),(x0[i+1],y0[j+1]),(x0[i],y0[j+1])])
                sect = ring.intersection(square)
                if len(sect.bounds)>0:
                    pipetest[round(square.bounds[1]/self.dy)][round(square.bounds[0]/self.dx)] += sect.length
        self.ngpipe = pipetest
        
        m.readshapefile('/Users/oulongwen/Downloads/PetroleumProduct_Pipelines_US_EIA/PetroleumProduct_Pipelines_US_Nov2014','diesel')
        pipetest = np.zeros((self.nrows,self.ncols))
        self.total_dieselpipe = 0
        for item in m.diesel:
            ring = LineString(item)
            self.total_dieselpipe += ring.length
            for code in list(set(self.idx.intersection(ring.bounds))):    # Remove duplicate values in idx.intersection results
                i,j = self.id_dict[code]
                square = Polygon([(x0[i],y0[j]),(x0[i+1],y0[j]),(x0[i+1],y0[j+1]),(x0[i],y0[j+1])])
                sect = ring.intersection(square)
                if len(sect.bounds)>0:
                    pipetest[round(square.bounds[1]/self.dy)][round(square.bounds[0]/self.dx)] += sect.length
        self.dieselpipe = pipetest
        return self
 

    def get_path(self, n0, n1):
        """If n0 and n1 are connected nodes in the graph, this function
        return an array of point coordinates along the road linking
        these two nodes."""
        return np.array(json.loads(self.sg[n0][n1]['Json'])['coordinates'])

    
    # def geocalc(lat0, lon0, lat1, lon1):
    def geocalc(self, lon0,lat0,lon1,lat1):
        """Return the distance (in km) between two points in 
        geographical coordinates."""
        EARTH_R = 6372.8
        lat0 = np.radians(lat0)
        lon0 = np.radians(lon0)
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        dlon = lon0 - lon1
        y = np.sqrt(
            (np.cos(lat1) * np.sin(dlon)) ** 2
             + (np.cos(lat0) * np.sin(lat1) 
             - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
        x = np.sin(lat0) * np.sin(lat1) + \
            np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
        c = np.arctan2(y, x)
        return EARTH_R * c

    def get_path_length(self, path):
        return np.sum(self.geocalc(path[1:,0], path[1:,1],
                              path[:-1,0], path[:-1,1]))

    def shortest(self, pos0, pos1):

        # Compute the length of the road segments.
        for n0, n1 in self.sg.edges_iter():
            path = self.get_path(n0, n1)
            distance = self.get_path_length(path)
            self.sg.edge[n0][n1]['distance'] = distance

        nodes = np.array(self.sg.nodes())
    # Get the closest nodes in the graph.
        pos0_i = np.argmin(np.sum((nodes - pos0)**2, axis=1))
        pos1_i = np.argmin(np.sum((nodes - pos1)**2, axis=1))

    # Compute the shortest path.
        path = nx.shortest_path(self.sg, 
                                source=tuple(nodes[pos0_i]), 
                                target=tuple(nodes[pos1_i]),
                                weight='distance')
        dis = 0
        for i in range(len(path)-1):
            dis += self.sg.edge[path[i]][path[i+1]]['distance']
        return path, dis

    
    def proset(self, sup, dem, k = 10, n = 10):
        suplist = list(zip(sup.lon,sup.lat,sup.prod,sup.x,sup.y))
        self.suplist = sorted(suplist,key = lambda x: x[2], reverse = True)[0:k]
        demlist = list(zip(dem.lon,dem.lat,dem.prod,dem.x,dem.y))
        self.demlist = sorted(demlist,key = lambda x: x[2], reverse = True)[0:n]
        slon,slat,supply,sx,sy = zip(*self.suplist)
        dlon,dlat,demand,dx,dy = zip(*self.demlist)
        self.S = np.array(supply)
        stotal = sum(supply)
        self.D = stotal * np.array(demand)/sum(demand)/1.01
        # d = [np.sqrt((x1-x2)**2+(y1-y2)**2) for x1, y1 in zip(sx,sy) for x2,y2 in zip(dx,dy)]
        
        self.d = []
        self.paths = []
        for lon0, lat0 in zip(slon, slat):
            for lon1, lat1 in zip(dlon, dlat):
                pos0 = (lon0, lat0)
                pos1 = (lon1, lat1)
                path, distance = self.shortest(pos0, pos1)
                self.d.append(distance)
                self.paths.append(path)
        #self.d = np.array(d)
        #self.d2 = [[(x1,y1),(x2,y2)] for x1,y1 in zip(sx,sy) for x2,y2 in zip(dx,dy)]
        # A1 = np.zeros((m+n, m*n))
        for i in range(k):
            A1 = spmatrix(1., [0]*n, range(i*n, (i+1)*n), (1, k*n))
            if i == 0:
                A0 = A1
            else:
                A0 = sparse([A0, A1])
        A2 = spmatrix(-1., range(n), range(n))
        A1 = sparse([[A2]]*k)
        A2 = spmatrix(-1., range(k*n), range(k*n))
        A1 = sparse([A0, A1, A2])

        b = np.hstack((self.S,-self.D,np.zeros(k*n)))
        d1 = cvxopt.matrix(self.d)
        b1 = cvxopt.matrix(b)
        return d1, A1, b1
        
    def transemi(self, s, d, k = 10, n = 10):
        
        m = self.mapgen
        x0 = self.x[0,:]
        y0 = self.y[:,0]
        c,G,h = self.proset(s,d,k,n)
        self.sols = solvers.lp(c,G,h,solver ='glpk')
        self.tranemi = np.zeros((self.nrows,self.ncols))
        self.totaltrans = 0
        for index,x in enumerate(np.array(self.sols['x'])):
            if x > .01 and self.d[index] > 0:
                temlon, temlat = zip(*self.paths[index])
                temx, temy = m(temlon, temlat)
                ring = LineString(zip(temx,temy))
                #ring = LineString(self.paths[index])
                self.totaltrans += ring.length * x
                for code in list(set(self.idx.intersection(ring.bounds))):
                    i,j = self.id_dict[code]
                    square = Polygon([(x0[i],y0[j]),(x0[i+1],y0[j]),(x0[i+1],y0[j+1]),(x0[i],y0[j+1])])
                    sect = ring.intersection(square)
                    if len(sect.bounds)>0:
                        self.tranemi[round(square.bounds[1]/self.dy)][round(square.bounds[0]/self.dx)] += sect.length * x
        return self.tranemi