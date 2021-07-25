"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ priImports ================================ #
import sys
import os
import platform
from pprint import pprint
import time

import auxLib as ax


# ================================ secImports ================================ #
import numpy as np
import networkx as nx
from map.map import MAP, load, showMap, loadReal
# from map.map_cyclic import MAP, load, showMap, loadReal

# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
from parameters import SEED, MAX_BINARY_LENGTH, TINY, wResource, capRatio, HORIZON, TOTAL_VESSEL, MAP_ID, wDelay, resMin, resMax, vSize, totCapRatio, T_MIN, T_MAX, LUCAS_CP_START, succProb, MODES, ARR_HORIZON, LOAD_MODEL, OLD_HORIZON, GRANULARITY, INSTANCE, BUFFER_MIN, BUFFER_MAX, AGENT_NAME
import math

from collections import Counter
from multiModArrival.multiModArrival import mmArrivalMulti
import random
import pdb
import rlcompleter
from auxLib import average, loadDataStr

# ------- Maps ------ #
map_id = MAP_ID
mp = MAP(map_id)
totalZONES = mp.totalZONES
dummyZONES = mp.dummyZONES
termZONES = mp.termZONES
planningZONES = mp.planningZones
planningTermZones = []
planningTermZones.extend(planningZONES)
planningTermZones.extend(termZONES)
totalResource = totalZONES
zGraph = mp.zGraph

np.random.seed(SEED)
listHORIZON = [ _ for _ in range(HORIZON)]
totalVESSELS = TOTAL_VESSEL


dummyNbrs = []
for zd in dummyZONES:
    z_nbr = nx.neighbors(mp.zGraph, zd)
    dummyNbrs.extend(z_nbr)
dummyNbrs = list(set(dummyNbrs))
planningNodummyZones = list(set(planningZONES)-set(dummyNbrs))
planningNodummyNbrZones = list(set(planningZONES)-set(dummyNbrs))

termNbrs = []
for z in planningZONES:
    z_nbrs = nx.neighbors(mp.zGraph, z)
    for zt in termZONES:
        if zt in z_nbrs:
            termNbrs.append(z)

class Zone:
    def __init__(self, id, nbr):
        self.id = id
        self.nbr = nbr

class ProblemInstance:
    def __init__(self):
        self.Zones = self._getZones()
        if map_id == -1:
            self.map = loadReal()
        elif map_id == -2:
            self.map = None
        # else:
        #     self.map = load("map/"+str(map_id)+".txt")
        self.mapID = map_id
        # print self.map
        # showMap(self.map)
        # exit()

    def _getZones(self):

        self.zGraph = mp.zGraph
        temp = []
        for i in range(totalZONES):
            temp.append(Zone(i, nx.neighbors(self.zGraph, i)))
        return temp

# -------------------------------------------- #
dirName = str(map_id)+"_"+str(wDelay)+"_"+str(wResource)+"_"+str(capRatio)+"_"+str(HORIZON)+"_"+str(totalVESSELS)+"_"+AGENT_NAME
os.system("mkdir ./log/")
os.system("mkdir ./log/"+dirName)

# -------- T_Min_Max ---------- #
maxTmax = 0
minTmin = 1e3
if "_" in str(MAP_ID):
    T_min_max = loadDataStr("./synData/"+str(MAP_ID)+"/"+str(INSTANCE)+"/T_min_max")
    for z in dummyNbrs:
        for zp in nx.neighbors(mp.zGraph, z):
            T_min_max[z][zp][0] = BUFFER_MIN
            if LOAD_MODEL:
                T_min_max[z][zp][1] = BUFFER_MAX
            else:
                T_min_max[z][zp][1] = BUFFER_MAX
else:
    T_min_max = np.array([[(T_MIN, T_MAX) for _ in range(totalZONES)] for _ in range(totalZONES)])
    for z in dummyNbrs:
        for zp in nx.neighbors(mp.zGraph, z):
            T_min_max[z][zp][0] = BUFFER_MIN
            if LOAD_MODEL:
                T_min_max[z][zp][1] = BUFFER_MAX
            else:
                T_min_max[z][zp][1] = BUFFER_MAX
for z in planningZONES:
    for zp in nx.neighbors(mp.zGraph, z):
        if minTmin > T_min_max[z][zp][0]:
            minTmin = T_min_max[z][zp][0]
        if maxTmax < T_min_max[z][zp][1]:
            maxTmax = T_min_max[z][zp][1]

# ---------- Arrival Time
arrivalTimeSynth = {}
arrivalTimeLucas = {}
def getFailVessel(tmpSuccVessel):
    totalHorizon = HORIZON
    tmpSuccVessel.append(0)
    tmpPop = sum(tmpSuccVessel)
    tmpFailVessel = [0]
    for tt in range(1, totalHorizon + 1):
        tmpFail = tmpPop - sum(tmpSuccVessel[1:tt+1])
        tmpFailVessel.append(tmpFail)
    return tmpSuccVessel, tmpFailVessel
if "_" in str(MAP_ID):
    arrivalTimeLucas = loadDataStr("./synData/"+str(MAP_ID)+"/"+str(INSTANCE)+"/arrivalTime_"+str(TOTAL_VESSEL))
else:
    tmpArrivalTime = {}
    arr = mmArrivalMulti(len(dummyZONES), MODES, ARR_HORIZON, HORIZON, TOTAL_VESSEL, SEED)
    mmArrival = arr.getArrival()
    for dz in dummyZONES:
        dzID = dummyZONES.index(dz)
        tmpArrivalTime[dz] = {}
        # MultiModal Arrival Distribution
        tmpSuccVessel, tmpFailVessel = getFailVessel(mmArrival[dzID])
        count = -1
        tmpArrivalTime[dz] = {}
        tmpArrivalTime[dz]['succ'] = {}
        tmpArrivalTime[dz]['fail'] = {}
        for tt in range(HORIZON):
            count += 1
            if tmpSuccVessel[count] <> 0:
                tmpArrivalTime[dz]['succ'][tt] = tmpSuccVessel[count]
                tmpArrivalTime[dz]['fail'][tt] = tmpFailVessel[count]
    arrivalTimeSynth = tmpArrivalTime

# ------ Resource Capacity-------- #
if "_" in str(MAP_ID):
    rCap = loadDataStr("./synData/" + str(MAP_ID) + "/" + str(INSTANCE) + "/rCap")
    # print rCap
    rCap = map(lambda v : int(math.ceil(v * capRatio)), rCap)
    # print rCap
    # pdb.set_trace()
    for z in dummyNbrs:
        rCap[z] = int(1e4)
else:
    rCap = map(lambda v : int(totCapRatio * totalVESSELS), [ i for i in range(totalResource)])
    rCap = map(lambda v : int(v * capRatio), rCap)
    for z in dummyNbrs:
        rCap[z] = int(1e4)
rCon = np.zeros((totalZONES, totalResource))
for z in planningZONES:
    for r in range(totalResource):
        if z == r:
            rCon[z][r] = vSize

# -------------------------------------------------------------- #
# --------- For Masking ------ #
oneHot_Size = 1 #totalVESSELS + 1
Mask = np.array([[0 for _ in range(totalZONES*oneHot_Size)] for _ in range(totalZONES)])
for z in range(totalZONES):
    Mask[z][oneHot_Size * z:z * oneHot_Size + oneHot_Size] = 1
    for z_nbr in nx.neighbors(mp.zGraph, z):
        Mask[z][oneHot_Size*z_nbr:z_nbr*oneHot_Size+oneHot_Size] = 1
for z in termNbrs:
    for zt in termZONES:
        if zt in nx.neighbors(mp.zGraph, z):
            Mask[z][zt] = 0
# ----------- Pzz ----------- #
# Pzz_mat = np.random.rand(totalZONES, totalZONES)
Pzz_mat = np.ones((totalZONES, totalZONES))
for z in range(totalZONES):
    for z_next in range(totalZONES):
        if z_next in dummyZONES:
            Pzz_mat[z][z_next] = 0 #TINY

        if z_next not in nx.neighbors(mp.zGraph, z):
            Pzz_mat[z][z_next] = 0 #TINY

# ---- Normalize ---- #
for z in range(totalZONES):
    for z_next in range(totalZONES):
        if np.sum(Pzz_mat[z]) > 0:
            Pzz_mat[z] = Pzz_mat[z]/np.sum(Pzz_mat[z])

def Pzz(z, z_next, zGraph):

    """
    This function models P(z_next | z) probability of moving to z_next given current zone is z
    :param z: Current Zone
    :param z_next: Next Zone
    :return: P(z_next | z)
    """
    return Pzz_mat[z][z_next]

# ---------------------------- #


# ----------- CP Baseline ------------ #
class cpData:

    def __init__(self, id):

        # --------- Processing Time --------- #
        self.proccTime = {}
        fname = "./map/lucas/" + id + "/" + str(TOTAL_VESSEL) + "/zoneGraph_" + id + "_" + str(TOTAL_VESSEL) + ".txt"
        with open(fname) as f:
            for line in f:
                if "processTime" in line:
                    # ----------- process Time
                    tmp = line.split(":")
                    t1 = tmp[0].split(" ")[1]
                    seqID = int(t1.strip())
                    self.proccTime[seqID] = {}
                    tmp = tmp[1]
                    tmp = tmp.split("#")
                    tmp = tmp[1:len(tmp)]
                    tmp = map(lambda v: v.strip(), tmp)
                    tmp3 = {}
                    for item in tmp:
                        tmp2 = item.split(",")
                        tmin = int(math.ceil(float((tmp2[1]))))
                        tmax = int(math.floor(float((tmp2[2]))))
                        tmp3[int(tmp2[0].strip())] = [tmin*GRANULARITY, tmax*GRANULARITY]
                    self.proccTime[seqID] = tmp3

        # ------------ Release Time -------- #
        fname = "./map/lucas/" + id + "/" + str(TOTAL_VESSEL) + "/"+id + "_" + str(TOTAL_VESSEL) + ".txt"
        with open(fname) as f:
            for i, line in enumerate(f):
                if i == 4:
                    t1 = line.split(" ")
                    t1 = filter(lambda x: x <>"\n", t1)
                    t1 = map(lambda x : float(x.strip()), t1)
                    self.releaseTime = list(t1)

        # ------------ Resource Cap -------- #
        fname = "./map/lucas/" + id + "/" + str(TOTAL_VESSEL) + "/" + id + "_" + str(TOTAL_VESSEL) + ".txt"
        with open(fname) as f:
            for i, line in enumerate(f):
                if i == 7:
                    t1 = line.split(" ")
                    t1 = filter(lambda x: x <> "\n", t1)
                    t1 = map(lambda x: int(x.strip()), t1)
                    self.resCap = list(t1)
        self.numRes = len(self.resCap)

        # ------------ Number of Sequence -------- #
        fname = "./map/lucas/" + id + "/" + str(TOTAL_VESSEL) + "/" + id + "_" + str(TOTAL_VESSEL) + ".txt"
        with open(fname) as f:
            for i, line in enumerate(f):
                if i == 2:
                    t1 = line.split(" : ")
                    t1 = int(t1[1].strip())
                    self.numSeq = t1

        # ------------ numAct ------------ #
        self.numActs = len(self.proccTime[0])
        # ------------ Res Con ------------ #
        self.rCon = np.zeros((self.numSeq, self.numActs, self.numRes))
        for i in range(self.numSeq):
            a = -1
            for ar in self.proccTime[i]:
                a += 1
                for r in range(self.numRes):
                    if ar == r:
                        self.rCon[i][a][r] = 1


class cpDataSynth:

    def __init__(self, id, k):

        self.Samples = k

        # --------- Processing Time --------- #
        self.start = []
        tmpProccTime = {}
        tmpStart = {}
        vID = 0
        for v in range(TOTAL_VESSEL):
            self.start.append(random.sample(dummyNbrs, 1)[0])
        self.numActs = len(planningZONES)
        self.proccTime = {}
        self.trajectory = {}
        for k in range(self.Samples):
            self.proccTime[k] = {}
            self.trajectory[k] = {}
            for v in range(TOTAL_VESSEL):
                self.proccTime[k][v] = {}
                self.trajectory[k][v] = []
                z = self.start[v]
                for a in range(0, self.numActs):
                    z_next = nx.neighbors(mp.zGraph, z)
                    pBucket = map(lambda zp : Pzz(z, zp, mp.zGraph), z_next)
                    zp_id = np.random.multinomial(1, pBucket)
                    zp_id = np.where(zp_id == 1)[0][0]
                    zp = z_next[zp_id]
                    self.proccTime[k][v][z] = [T_min_max[z][zp][0], T_min_max[z][zp][1]]
                    self.trajectory[k][v].append(z)
                    z = zp

        # ------------ Release Time -------- #
        arrTmpSucc = []
        for dz in dummyZONES:
            for tt in range(HORIZON+1):
                if tt in arrivalTimeSynth[dz]['succ']:
                    totVess = arrivalTimeSynth[dz]['succ'][tt]
                    for ttt in range(totVess):
                        arrTmpSucc.append(tt)
        self.releaseTime = arrTmpSucc

        # ------------ Resource Cap -------- #
        self.resCap = rCap
        # self.numRes = len(self.resCap) - len(termZONES) - len(dummyZONES)
        self.numRes = totalZONES
        # ------------ Res Con ------------ #
        self.rCon = np.zeros((self.Samples, TOTAL_VESSEL, self.numActs, self.numRes))
        for k in range(self.Samples):
            for i in range(TOTAL_VESSEL):
                a = -1
                for ar in self.proccTime[k][i]:
                    a += 1
                    for r in range(self.numRes):
                        if ar == r:
                            self.rCon[k][i][a][r] = 1


class cpDataSAA:

    def __init__(self, id, k):

        self.Samples = k
        # -------- T_min_max ------ #


        # --------- Processing Time --------- #
        tmpProccTime = {}
        fname = "./map/lucas/" + id + "/" + str(TOTAL_VESSEL) + "/zoneGraph_" + id + "_" + str(TOTAL_VESSEL) + ".txt"
        tmpStart = {}
        vID = 0
        with open(fname) as f:
            for line in f:
                if "processTime" in line:
                    tmpStart[vID] = []
                    # ----------- process Time
                    tmp = line.split(":")
                    t1 = tmp[0].split(" ")[1]
                    seqID = int(t1.strip())
                    tmpProccTime[seqID] = {}
                    tmp = tmp[1]
                    tmp = tmp.split("#")
                    tmp = tmp[1:len(tmp)]
                    tmp = map(lambda v: v.strip(), tmp)
                    tmp3 = {}
                    for item in tmp:
                        tmp2 = item.split(",")
                        tmpStart[vID].append(int(tmp2[0].strip()))
                        tmin = int(math.ceil(float((tmp2[1]))))
                        tmax = int(math.floor(float((tmp2[2]))))
                        tmp3[int(tmp2[0].strip())] = [tmin*GRANULARITY, tmax*GRANULARITY]
                    tmpProccTime[seqID] = tmp3
                    vID += 1
        # ------ starting zone
        self.start = []
        for v in range(TOTAL_VESSEL):
            self.start.append(tmpStart[v][0])
        self.numActs = len(tmpProccTime[0])
        self.proccTime = {}
        self.trajectory = {}
        for k in range(self.Samples):
            self.proccTime[k] = {}
            self.trajectory[k] = {}
            for v in range(TOTAL_VESSEL):
                self.proccTime[k][v] = {}
                self.trajectory[k][v] = []
                z = self.start[v]
                for a in range(0, self.numActs):
                    z_next = nx.neighbors(mp.zGraph, z)
                    pBucket = map(lambda zp : Pzz(z, zp, mp.zGraph), z_next)
                    zp_id = np.random.multinomial(1, pBucket)
                    zp_id = np.where(zp_id == 1)[0][0]
                    zp = z_next[zp_id]
                    self.proccTime[k][v][z] = [T_min_max[z][zp][0], T_min_max[z][zp][1]]
                    self.trajectory[k][v].append(z)
                    z = zp

        # ------------ Release Time -------- #
        p = 1
        arrTmpSucc = []
        for dz in dummyZONES:
            for tt in range(HORIZON+1):
                if tt in arrivalTimeLucas[dz][p]['succ']:
                    totVess = arrivalTimeLucas[dz][p]['succ'][tt]
                    for ttt in range(totVess):
                        arrTmpSucc.append(tt)
        self.releaseTime = arrTmpSucc

        # fname = "./map/lucas/" + id + "/" + str(TOTAL_VESSEL) + "/"+id + "_" + str(TOTAL_VESSEL) + ".txt"
        # with open(fname) as f:
        #     for i, line in enumerate(f):
        #         if i == 4:
        #             t1 = line.split(" ")
        #             t1 = filter(lambda x: x <>"\n", t1)
        #             t1 = map(lambda x : float(x.strip()), t1)
        #             self.releaseTime = list(t1)

        # ------------ Resource Cap -------- #
        self.resCap = rCap
        self.numRes = len(self.resCap) - len(termZONES) - len(dummyZONES)

        # ------------ Res Con ------------ #
        self.rCon = np.zeros((self.Samples, TOTAL_VESSEL, self.numActs, self.numRes))
        for k in range(self.Samples):
            for i in range(TOTAL_VESSEL):
                a = -1
                for ar in self.proccTime[k][i]:
                    a += 1
                    for r in range(self.numRes):
                        if ar == r:
                            self.rCon[k][i][a][r] = 1




# -------------------------- #