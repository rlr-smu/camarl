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
from auxLib import average

# ================================ secImports ================================ #
import numpy as np
import networkx as nx
from matplotlib import patches
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
import seaborn
from helperLib import updateZoneCount, logic, runningAvg, p_nbr, cost_t_nt_z, getMaxReward
from parameters import xAxes, BATCH_SIZE, succProb, SEED, MAP_ID
import scipy.stats
from data import totalZONES, HORIZON, dummyZONES, termZONES, totalVESSELS, Pzz_mat, planningZONES, T_min_max, arrivalTimeLucas, arrivalTimeSynth, dummyNbrs, planningTermZones, planningNodummyNbrZones
import shapefile as shp
from scipy.special import comb
import pdb
import rlcompleter

# ============================================================================ #

# --------------------- Variables ------------------------------ #
np.random.seed(SEED)
ppath = os.getcwd() + "/"  # Project Path Location

# -------------------------------------------------------------- #

class countTable:

    def __init__(self):
        self.nt_zt = np.zeros((totalZONES, HORIZON + 1))   # n_t(z, \tau = t)
        self.nt_zz = np.zeros((totalZONES, totalZONES, HORIZON + 1)) # n_t(z, z', \tau > t)
        self.nt_zt_zt = np.zeros((totalZONES, HORIZON + 1, totalZONES, HORIZON + 1)) # n_t(z, \tau, z', \tau')
        self.nt_z = np.zeros(totalZONES)   # n_t(z)
        self.ntz_zt = np.zeros((totalZONES, totalZONES, HORIZON+1))
        self.nt_ztz = np.zeros((totalZONES, totalZONES)) # n_t(z,z')


class Maritime:

    def __init__(self, instance):

        self.instance = instance
        self.renderFlag = False
        self.zCount = None
        self.zCountList = [[] for _ in range(BATCH_SIZE)]
        self.zCountList_1hot = [[] for _ in range(BATCH_SIZE)]
        self.cT = countTable
        self.t = None
        self.reward = []
        self.rAvg = 0
        self.rAvgList = []
        self.beta_all = np.zeros((HORIZON+2, totalZONES, totalZONES))
        self.maxReward = getMaxReward()
        # self.notReached = np.zeros((totalZONES, PERIOD+1, HORIZON))
        self.arrivalTime = {}
        if "_" in str(MAP_ID):
            self.arrivalTime = arrivalTimeLucas
        else:
            self.arrivalTime = arrivalTimeSynth

    def step(self, t, cT, beta):


        self.beta_all[t] = beta
        self.cT = cT
        self.t = t
        '''Required only during rendering'''
        # self.zCount = updateZoneCount(self.t, self.cT, self.instance.zGraph)
        # self.reward.append(self.rt)  # Required for rendering

        # ---------- Rewards ----------- #
        self.rt_z = self._atomicReward(self.cT.nt_z)
        self.rt = np.sum(self.rt_z * self.cT.nt_z)

        # Normalize the reward
        self.rt_z = self.rt_z / self.maxReward

        # -------------------------------------------- #
        self.cT = self._sample(t, self.cT)

        return self.rt, self.rt_z, self.cT

    def _atomicReward(self, nt_z):

        tmp = np.zeros((totalZONES))
        for z in planningZONES:
            tmp[z] = -1 * cost_t_nt_z(z, nt_z[z])
        return tmp

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
                tMin = T_min_max[z][zp][0]
                tMax = T_min_max[z][zp][1]
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

    def _sampleBCKUP(self, t, cT):

        zGraph = self.instance.zGraph
        # ------- Sample z' --------- #
        # sample n_t(z, \tau=t, z') from n_t(z, \tau=t)
        ztz = np.zeros((totalZONES, totalZONES))
        for z in planningZONES:
            pBucket = Pzz_mat[z]
            if sum(pBucket) > 0:
                ztz[z] = np.random.multinomial(cT.nt_zt[z][t], pBucket)

        # ------- Sample \tau' ------ #
        # --- New
        # sample nt(z, t, z', t+k ) from nt(z, t, z')
        notReached = np.zeros(totalZONES)
        for z in planningZONES:
            for zp in nx.neighbors(zGraph, z):
                tMin = T_min_max[z][zp][0]
                tMax = T_min_max[z][zp][1]
                n = tMax - tMin
                beta = self.beta_all[t][z][zp]
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
