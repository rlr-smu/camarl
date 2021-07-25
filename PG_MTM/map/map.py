"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 10 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ secImports ================================ #

import sys
import os
import platform
from pprint import pprint
import time

# o = platform.system()
# if o == "Linux":
#     d = platform.dist()
#     if d[0] == "debian":
#         sys.path.append("/home/james/Codes/python")
#     if d[0] == "centos":
#         sys.path.append("/home/arambamjs.2016/projects")
#     if d[0] == "redhat":
#         sys.path.append("/home/arambamjs/projects")
# if o == "Darwin":
#     sys.path.append("/Users/james/Codes/python")
# import auxlib.auxLib as ax

import auxLib as ax
from auxLib import average
# ================================ priImports ================================ #
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
import seaborn
import numpy as np
import networkx as nx
import shapefile as shp
import math
from collections import Counter
# sys.path.append("../")
from parameters import GRANULARITY, SEED, TOTAL_VESSEL, LUCAS_CP_START


np.random.seed(SEED)

# ============================================================================ #

# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location

# -------------------------------------------------------------- #
class TSS:

    def __init__(self):
        self.true_zones = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 53, 54]
        self.true_edges = [(1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (10, 11), (11, 12), (12, 13), (13, 14), (14, 53), (8, 53), (9, 53), (53, 21), (53, 22), (53, 20), (20, 54), (21, 54), (22, 54), (54, 18), (54, 15), (18, 19), (15, 16), (19, 23), (16, 17)]

        self.zoneHash = {}
        map(lambda i: self.zoneHash.update({self.true_zones[i]: i}), [_ for _ in range(len(self.true_zones))])

        self.edges = []
        for e in self.true_edges:
            self.edges.append((self.zoneHash[e[0]], self.zoneHash[e[1]]))


class MAP:

    def __init__(self, id):

        self.arrivalTime = {}
        if id == 0:
            self.totalZONES = 4
            self.dummyZONES = [0]
            self.termZONES = [3]

        elif id == 1:
            self.totalZONES = 6
            self.dummyZONES = [0]
            self.termZONES = [5]
            # self.planningZones = list(set([_ for _ in range(self.totalZONES)]) - set(self.dummyZONES) - set(self.termZONES))

        elif id == 2:
            self.totalZONES = 19
            self.dummyZONES = [12, 13]
            #self.dummyZONES = [13]
            self.termZONES = [9, 10, 11]
            # self.planningZones = list(set([_ for _ in range(self.totalZONES)]) - set(self.dummyZONES) - set(self.termZONES))

        elif id == 3:
            self.totalZONES = 9
            self.dummyZONES = [0]
            self.termZONES = [6, 7, 8]
            # self.planningZones = list(set([_ for _ in range(self.totalZONES)]) - set(self.dummyZONES) - set(self.termZONES))

        elif id == -1:
            self.totalZONES = 25
            self.dummyZONES = [0, 9]
            self.termZONES = [16, 22]
            # self.planningZones = list(set([_ for _ in range(self.totalZONES)]) - set(self.dummyZONES) - set(self.termZONES))

        # if id == -2:
        elif "_" in str(id):
            t_min_max = {}
            act_zone = {}
            act_zone_rev = {}
            # fname = "./map/lucas/"+id+"/"+str(TOTAL_VESSEL)+"/zoneGraph_"+id+"_"+str(TOTAL_VESSEL)+".txt"
            fname = "./map/lucas/"+id+"/zoneGraph_"+id+".txt"

            # fname = "./map/lucas/zoneGraph_3_2_1000.txt"
            with open(fname) as f:
                for line in f:
                    if "dummyZones" in line:
                        tmp = line.split(":")
                        tmp = tmp[1]
                        tmp = tmp.split(",")
                        tmp = map(lambda x : int(x.strip()), tmp)
                        self.dummyZONES = []
                        self.dummyZONES.extend(tmp)
                    if "termZones" in line:
                        tmp = line.split(":")
                        tmp = tmp[1]
                        tmp = tmp.split(",")
                        tmp = map(lambda x: int(x.strip()), tmp)
                        self.termZONES = []
                        self.termZONES.extend(tmp)
                    if "nodes" in line:
                        tmp = line.split(":")
                        tmp = tmp[1]
                        tmp = tmp.split(",")
                        tmp = map(lambda x: int(x.strip()), tmp)
                        self.totalZONES = len(tmp)
                    if "act_zone" in line:
                        tmp = line.split(":")
                        act = int(tmp[1])
                        tmp = tmp[2]
                        tmp = tmp.split(",")
                        tmp = tmp[0:len(tmp)-1]
                        tmp = map(lambda v : int(v), tmp)
                        act_zone_rev[act] = tmp
                        for z in tmp:
                            act_zone[z] = act
                    # if "processTime" in line:
                    #     # ----------- Travel Time
                    #     tmp = line.split(":")
                    #     tmp = tmp[1]
                    #     tmp = tmp.split("#")
                    #     tmp = tmp[1:len(tmp)]
                    #     tmp = map(lambda v : v.strip(), tmp)
                    #     for item in tmp:
                    #         tmp2 = item.split(",")
                    #         if int(tmp2[0]) not in t_min_max:
                    #             t_min_max[int(tmp2[0])] = {}
                    #             t_min_max[int(tmp2[0])]['Min'] = []
                    #             t_min_max[int(tmp2[0])]['Max'] = []
                    #         tmin = int(math.ceil(float((tmp2[1]))))
                    #         tmax = int(math.floor(float((tmp2[2]))))
                    #         t_min_max[int(tmp2[0])]['Min'].append(tmin * GRANULARITY)
                    #         t_min_max[int(tmp2[0])]['Max'].append(tmax * GRANULARITY)

            # for z in range(self.totalZONES):
            #     if z not in t_min_max and z not in self.dummyZONES and z not in self.termZONES:
            #         actv = act_zone[z]
            #         tmp = filter(lambda v: v <> z, act_zone_rev[actv])
            #         for z2 in tmp:
            #             if z2 in t_min_max:
            #                 t_min_max[z] = t_min_max[z2]

            # ------ Average Travel Time ------- #
            # self.T_Min_Max = {}
            # for z in t_min_max:
            #     t_min_max[z]['Avg'] = (int(math.ceil(average(t_min_max[z]['Min']))), int(math.floor(average(t_min_max[z]['Max']))))
            #     self.T_Min_Max[z] = t_min_max[z]['Avg']

            # --------- Arrival Time
            # if LUCAS_CP_START:
            #     tmpG = self._createZoneGraph(id)
            #     dum2pla = {}
            #     for (i, j) in tmpG.edges():
            #         if i in self.dummyZONES:
            #             dum2pla[i] = j
            #     tmpArr2 = {}
            #     fname = "./map/lucas/"+id+"/"+str(TOTAL_VESSEL)+"/"+"sch_"+str(TOTAL_VESSEL)+".txt"
            #     count = 0
            #     for dz in self.dummyZONES:
            #         tmpArr2[dz] = []
            #         with open(fname) as f:
            #             for line in f:
            #                 if " "+str(dum2pla[dz])+" |" in line:
            #                     tmp = line.split("|")
            #                     tmpArr2[dz].append(int(math.ceil(float((tmp[1].strip())))))
            #                     count += 1
            #         self.arrivalTime[dz] = tmpArr2[dz]

                # self.arrivalTime[dz] = dict(Counter(tmpArr2[dz]))
            # --------- Resource Capacity
            # Resource Capacity is used from Tmin policy implemented in computeCapacity.py which generates a file capacity.txt in /lib/id+totalVessels/
            # fname = "./map/lucas/"+id+"/"+str(TOTAL_VESSEL)+"/"+id+"_"+str(TOTAL_VESSEL)+".txt"
            # with open(fname) as f:
            #     for i, line in enumerate(f):
            #         if i == 7:
            #             tmp = line.strip().split(" ")
            #             tmp = map(lambda x : int(x), tmp)
            #             self.rCap = tmp
            #             for z in self.dummyZONES:
            #                 self.rCap.append(0)
            #             for z in self.termZONES:
            #                 self.rCap.append(0)


        self.planningZones = list(set([_ for _ in range(self.totalZONES)]) - set(self.dummyZONES) - set(self.termZONES))
        self.zGraph = self._createZoneGraph(id)


    def _createZoneGraph(self, id):

        G = nx.DiGraph()

        # ------ Graph for Map : 0 ----- #
        if id == 0:
            G.add_edge(0, 1)
            G.add_edge(1, 2)
            G.add_edge(2, 3)

        # ------ Graph for Map : 1 ----- #
        elif id == 1:
            G.add_edge(0, 1)
            G.add_edge(1, 2)
            G.add_edge(2, 3)
            G.add_edge(3, 4)
            G.add_edge(4, 5)

        # ------ Graph for Map : 2 ----- #
        elif id == 2:
            G.add_edge(13, 0)
            G.add_edge(0, 1)
            G.add_edge(1, 2)
            G.add_edge(2, 15)
            G.add_edge(15, 14)
            G.add_edge(14, 18)
            G.add_edge(12, 3)
            G.add_edge(3, 4)
            G.add_edge(4, 5)
            G.add_edge(5, 14)
            G.add_edge(14, 18)
            G.add_edge(18, 16)
            G.add_edge(16, 17)
            G.add_edge(17, 10)
            G.add_edge(16, 6)
            G.add_edge(6, 9)
            G.add_edge(16, 7)
            G.add_edge(7, 8)
            G.add_edge(8, 11)

        # ------ Graph for Map : 3 ----- #
        elif id == 3:
            G.add_edge(0, 1)
            G.add_edge(1, 2)
            G.add_edge(2, 3)
            G.add_edge(2, 4)
            G.add_edge(2, 5)
            G.add_edge(5, 7)
            G.add_edge(4, 6)
            G.add_edge(3, 8)

        # ----- Graph for Map : -1 (Real TSS Map) ----- #
        elif id == -1:
            tss = TSS()
            for e in tss.edges:
                G.add_edge(e[0], e[1])

        # ------ Graph for Lucas's synthetic data ---- #
        elif "_" in id:
            fname = "./map/lucas/"+id+"/zoneGraph_"+id+".txt"
            with open(fname) as f:
                for i, line in enumerate(f):
                    if i > 3 and "edge" in line:
                        tmp = line.split(",")
                        tmp = tmp[1:len(tmp)]
                        tmp = map(lambda x: int(x.strip()), tmp)
                        G.add_edge(tmp[0], tmp[1])
        return G


# -------------- Real Map -------------- #

def shpFile():

    def avg(x, y):
        x_avg = sum(x) / len(x)
        y_avg = sum(y) / len(y)
        return x_avg, y_avg

    def extractCoordinates(x, y):
        tmp = map(lambda i: (x[i], y[i]), [_ for _ in range(len(x))])
        return tmp

    instance = {}
    tss = TSS()
    # --------- TSS -------- #
    sf = shp.Reader("./map/TSS/zones.shp")
    fields = [field[0] for field in sf.fields[1:]]
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        atr = dict(zip(fields, shape.record))
        id = int(atr['id'])
        if id in tss.true_zones:
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            xa, ya = avg(x, y)
            instance[int(tss.zoneHash[id])] = [extractCoordinates(x, y), [(xmin, xmax), (ymin, ymax)], [xa, ya]]

    return instance

def loadReal():

    fig = shpFile()
    return fig

def load(map):

    coord = []
    fig = {}
    with open(map) as f:
        for line in f:
            tmp = line.split(";")
            for line in tmp:
                if line[0:5] == " draw" and "cycle" in line:
                    line = line[6:len(line)]
                    tmp2 = line.split("--")
                    tmp2 = tmp2[0:4]
                    coord.append(tmp2)
    pCount = 0
    for p in coord:
        tmp2 = []
        tmpx = []
        tmpy = []
        for c in p:
            tmp = c.split(",")
            x = float(tmp[0][1:len(tmp[0])])
            y = float(tmp[1][0:len(tmp[1])-1])
            tmp2.append((x, y))
            tmpx.append(x)
            tmpy.append(y)
        fig[pCount] = [tmp2, [(min(tmpx), max(tmpx)), (min(tmpy), max(tmpy))]]
        pCount += 1
    return fig

def showMap(map):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in map:
        p = Polygon(map[i][0])
        x, y = p.exterior.xy
        ax.plot(x, y, 'black')
        xt = np.average(x)
        yt = np.average(y)
        ax.annotate('Z' + str(i), xy=(xt, yt), xytext=(xt, yt))
    plt.show()

def main():

    map = load("1.txt")
    showMap(map)


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
