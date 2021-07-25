"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 12 Jul 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import auxLib as ax
from auxLib import dumpDataStr
import pdb
import rlcompleter
from multiModArrival.multiModArrival import mmArrivalMulti
from parameters import TOTAL_VESSEL, HORIZON, ARR_HORIZON, MODES, MAP_ID
from map.map import MAP, loadReal
import numpy as np
import networkx as nx
import argparse

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #
parse = argparse.ArgumentParser()
parse.add_argument("-i", "--instance", type = int, help="Number of Instances")
args = parse.parse_args()
INSTANCES = args.instance
TMIN = [2, 2]
TMAX = [10, 20]
rCapRange = [5, 25]

# ---------------------------- #
mp = MAP(MAP_ID)
totalZONES = mp.totalZONES
dummyZONES = mp.dummyZONES
totalVESSELS = TOTAL_VESSEL
planningZONES = mp.planningZones
termZONES = mp.termZONES

zGraph = mp.zGraph
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
# ============================================================================ #

class Zone:
    def __init__(self, id, nbr):
        self.id = id
        self.nbr = nbr

class countTable:

    def __init__(self):
        self.nt_zt = np.zeros((totalZONES, HORIZON + 1))   # n_t(z, \tau = t)
        self.nt_zz = np.zeros((totalZONES, totalZONES, HORIZON + 1)) # n_t(z, z', \tau > t)
        self.nt_zt_zt = np.zeros((totalZONES, HORIZON + 1, totalZONES, HORIZON + 1)) # n_t(z, \tau, z', \tau')
        self.nt_z = np.zeros(totalZONES)   # n_t(z)
        self.ntz_zt = np.zeros((totalZONES, totalZONES, HORIZON+1))
        self.nt_ztz = np.zeros((totalZONES, totalZONES)) # n_t(z,z')

class ProblemInstance:
    def __init__(self):
        self.Zones = self._getZones()
        if MAP_ID == -1:
            self.map = loadReal()
        elif MAP_ID == -2:
            self.map = None
        # else:
        #     self.map = load("map/"+str(map_id)+".txt")
        self.mapID = MAP_ID
        # print self.map
        # showMap(self.map)
        # exit()

    def _getZones(self):

        self.zGraph = mp.zGraph
        temp = []
        for i in range(totalZONES):
            temp.append(Zone(i, nx.neighbors(self.zGraph, i)))
        return temp

class Maritime:

    def __init__(self, instance, arrivalTime, T_min_max):

        self.instance = instance
        self.t = None
        self.arrivalTime = arrivalTime
        self.T_min_max = T_min_max
        self.beta_all = np.zeros((HORIZON+2, totalZONES, totalZONES))

    def step(self, t, cT, beta):

        self.beta_all[t] = beta
        self.cT = self._sample(t, cT)
        return self.cT

    def _sample(self, t, cT):

        zGraph = self.instance.zGraph
        # ------- Sample z' --------- #
        # sample n_t(z, \tau=t, z') from n_t(z, \tau=t)
        ztz = np.zeros((totalZONES, totalZONES))
        for z in planningZONES:
            pBucket = Pzz_mat[z]
            if sum(pBucket) > 0:
                ztz[z] = np.random.multinomial(cT.nt_zt[z][t], pBucket)

        # -------- update nt_ztz
        cT.nt_ztz = ztz

        # ------- Sample \tau' ------ #
        # --- New
        # sample nt(z, t, z', t+k ) from nt(z, t, z')
        notReached = np.zeros(totalZONES)
        for z in planningZONES:
            for zp in nx.neighbors(zGraph, z):
                tMin = self.T_min_max[z][zp][0]
                tMax = self.T_min_max[z][zp][1]
                n = tMax - tMin
                beta = self.beta_all[t][z][zp]
                # print t, z, zp, beta
                tmpSample = np.random.binomial(n, beta, int(ztz[z][zp]))
                countBuckets = []
                for c in range(n + 1):
                    countBuckets.append(np.count_nonzero(tmpSample == c))
                if len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1]) < len(countBuckets):
                    len_nt = len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1])
                    len_pB = len(countBuckets)
                    cT.nt_zt_zt[z][t][zp][t + tMin:t + tMax + 1] = countBuckets[0:len_nt]
                    if sum(countBuckets[len_nt:len_pB]) > 0:
                        notReached[z] += sum(countBuckets[len_nt:len_pB])

                else:
                    cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1] = countBuckets


        # -------- Update Count Tables for next system time ------- #
        for z in range(totalZONES):
            if z not in dummyZONES:
                # ------- cT.nt_zt (Newly Arrived Agents) ------- #
                cT.nt_zt[z][t+1] = sum(cT.nt_zz[:,z,t+1]) + sum(cT.nt_zt_zt[:,t,z,t+1]) + notReached[z]

                # ------- cT.nt_zz (Transiting Agents)------- #
                for zp in range(totalZONES):
                    cT.nt_zz[z][zp][t+2:] = cT.nt_zz[z][zp][t+2:] + cT.nt_zt_zt[z][t][zp][t+2:]

                # ------ cT.nt_z ( Total Agents present in zone z)------ #
                cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(totalZONES)]))
                # ------ cT.ntz_zt (used in fictitious)----- #
                cT.ntz_zt[z] = cT.nt_zt_zt[z][t]


        # ------- Update Dummy Zone Count ------ #
        for dz in dummyZONES:
            for tp in self.arrivalTime[dz]['succ']:
                if t + 1 == tp:
                    succVessel = self.arrivalTime[dz]['succ'][tp]
                    failVessel = self.arrivalTime[dz]['fail'][tp]
                    pz = nx.neighbors(zGraph, dz)[0]
                    cT.nt_zt[pz][t+1] += succVessel
                    cT.nt_z[pz] += succVessel
                    cT.nt_z[dz] = failVessel


        # ------- Update Terminal Zone Count -------- #
        for z in termZONES:
            cT.nt_zt[z][t + 1] += cT.nt_zt[z][t]
            cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(totalZONES)]))

        assert sum(cT.nt_z) == totalVESSELS, "Count Error : "+str(cT.nt_z)
        return  cT

def initMap(cT, instance, dummyZONES, totalVESSELS):

    nDummy = len(dummyZONES)
    for dz in dummyZONES:
        nbr = nx.neighbors(instance.zGraph, dz)
        if len(nbr) == 1:
            '''cT.nt_zz[dz][nbr[0]][1] to cT.nt_zz[dz][nbr[0]][0] for arrival rate'''
            # cT.nt_zz[dz][nbr[0]][1] = totalVESSELS / nDummy
            # cT.nt_z[dz] = totalVESSELS / nDummy
            cT.nt_zz[dz][nbr[0]][0] = totalVESSELS / nDummy
            cT.nt_z[dz] = totalVESSELS / nDummy

        else:
            print "Nbr of dummyZone > 1"
            exit()

    return cT

def generateArrivalTime(dPath, SEED):

    def getFailVessel(tmpSuccVessel):
        totalHorizon = HORIZON
        tmpSuccVessel.append(0)
        tmpPop = sum(tmpSuccVessel)
        tmpFailVessel = [0]
        for tt in range(1, totalHorizon + 1):
            tmpFail = tmpPop - sum(tmpSuccVessel[1:tt + 1])
            tmpFailVessel.append(tmpFail)
        return tmpSuccVessel, tmpFailVessel

    arr = mmArrivalMulti(len(dummyZONES), MODES, ARR_HORIZON, HORIZON, TOTAL_VESSEL, SEED)
    mmArrival = arr.getArrival()
    tmpArrivalTime = {}
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
    arrivalTime = tmpArrivalTime
    dumpDataStr(dPath+"/arrivalTime", arrivalTime)
    return arrivalTime

def generateTminTmax(dPath, SEED):
    np.random.seed(SEED)
    # ------- 1st Method
    T_min_max = np.array([[(-1, -1) for _ in range(totalZONES)] for _ in range(totalZONES)])
    for z in range(totalZONES):
        for zp in nx.neighbors(zGraph, z):
            tmin = np.random.randint(TMIN[0], TMIN[1]+1)
            tmax = np.random.randint(TMAX[0], TMAX[1]+1)
            T_min_max[z][zp][0] = tmin
            T_min_max[z][zp][1] = tmax
    print T_min_max
    # ------- 2nd Method
    '''
    T_min_max = np.array([[(-1, -1) for _ in range(totalZONES)] for _ in range(totalZONES)])
    for z in range(totalZONES):
        for zp in nx.neighbors(zGraph, z):
            tmin = np.random.randint(TMIN[0], TMIN[1])
            tmax = 2 * tmin
            T_min_max[z][zp][0] = tmin
            T_min_max[z][zp][1] = tmax
    '''
    dumpDataStr(dPath + "/T_min_max", T_min_max)
    return T_min_max

def generateResCap(dPath, SEED, arrivalTime, T_min_max):

    # ----- 1st Method
    np.random.seed(SEED)
    rCap = np.zeros(totalZONES)
    for z in range(totalZONES):
        rCap[z] = np.random.randint(rCapRange[0], rCapRange[1]+1)
    dumpDataStr(dPath + "/rCap", rCap)
    print rCap
    # ------ 2nd Method
    '''
    instance = ProblemInstance()
    env = Maritime(instance, arrivalTime, T_min_max)
    # --------- Buffers --------- #
    cT = countTable()
    cT = initMap(cT, instance, dummyZONES, totalVESSELS)
    count = np.zeros((HORIZON, totalZONES))
    for t in range(HORIZON):
        beta = np.zeros((totalZONES, totalZONES))
        count[t] = cT.nt_z
        cT_new = env.step(t, cT, beta)
        cT = cT_new
    rCap = np.zeros(totalZONES)
    for z in range(totalZONES):
        rCap[z] = int(np.max(count[:,z]))
    dumpDataStr(dPath + "/rCap", rCap)
    '''

def main():

    for i in range(1, INSTANCES+1):
        print ""
        print "Instance ", i
        os.system("mkdir ./synData/"+str(MAP_ID)+"_"+str(TOTAL_VESSEL))
        os.system("mkdir ./synData/" + str(MAP_ID)+"_"+str(TOTAL_VESSEL)+"/"+str(i))
        dPath = "./synData/" + str(MAP_ID)+"_"+str(TOTAL_VESSEL)+"/"+str(i)
        arrival = generateArrivalTime(dPath, i)
        tminmax = generateTminTmax(dPath, i)
        generateResCap(dPath, i, arrival, tminmax)


# =============================================================================== #

if __name__ == '__main__':
    main()
    