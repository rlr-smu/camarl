"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 14 Apr 2019
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys

sys.dont_write_bytecode = True
import numpy as np
import networkx as nx
from utils import cost_t_nt_z, getMaxReward
import scipy.stats
import pdb
import rlcompleter
import os
import pdb
import rlcompleter
import random


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #
ppath = os.getcwd() + "/"  # Project Path Location


# ============================================================================ #

class countTable:

    def __init__(self, totalZONES, HORIZON, NUM_OPTIONS):
        self.nt_zt = np.zeros((totalZONES, HORIZON + 1))   # n_t(z, \tau = t)
        self.nt_zz = np.zeros((totalZONES, totalZONES, HORIZON + 1)) # n_t(z, z', \tau > t)
        self.nt_zt_zt = np.zeros((totalZONES, HORIZON + 1, totalZONES, HORIZON + 1)) # n_t(z, \tau, z', \tau')
        self.nt_z = np.zeros(totalZONES)   # n_t(z)
        self.ntz_zt = np.zeros((totalZONES, totalZONES, HORIZON+1))
        self.nt_z_o = np.zeros((totalZONES, totalZONES, NUM_OPTIONS)) # n_t(z,z',\omega)
        self.nt_z_o_tau = np.zeros((totalZONES, totalZONES, NUM_OPTIONS, HORIZON+1)) # n_t(z,z',\omega, \tau)
        # self.nt_z_zp = np.zeros((totalZONES, totalZONES))

class Maritime:

    def __init__(self, data=None):

        np.random.seed(data.seed)
        random.seed(data.seed)

        self.map_id = data.map_id
        self.totalZONES = data.totalZONES
        self.dummyZONES = data.dummyZONES
        self.termZONES = data.termZONES
        self.Pzz_mat = data.Pzz_mat
        self.planningZONES = data.planningZONES
        self.T_min_max = data.T_min_max
        self.batch_size = data.batch_size
        self.zGraph = data.zGraph
        self.num_options = data.num_options
        self.renderFlag = False
        self.zCount = None
        self.zCountList = [[] for _ in range(self.batch_size)]
        self.zCountList_1hot = [[] for _ in range(self.batch_size)]
        self.cT = countTable
        self.t = None
        self.reward = []
        self.rAvg = 0
        self.rAvgList = []
        self.maxReward = getMaxReward()
        if "real" not in self.map_id:
            self.arr_seeds = data.arr_seeds
            self.total_vessels = data.total_vessels
            self.arrivalTimeDict = data.arrivalTimeDict
            self.HORIZON = data.HORIZON

        else:
            self.arrival_dist = data.arrival_dist
            self.initialCount = data.initialCount
            self.HORIZON = data.real_horizon

        self.vizData = {}


    def getFailVessel(self, tmpSuccVessel, total_horizon):
        totalHorizon = total_horizon
        tmpSuccVessel.append(0)
        tmpPop = sum(tmpSuccVessel)
        tmpFailVessel = [0]
        for tt in range(1, totalHorizon + 1):
            tmpFail = tmpPop - sum(tmpSuccVessel[1:tt + 1])
            tmpFailVessel.append(tmpFail)
        return tmpSuccVessel, tmpFailVessel

    def init(self):

        cT = countTable(self.totalZONES, self.HORIZON, self.num_options)
        if "real" in self.map_id:
            arr = []
            arr_count = 0
            i = 0
            for dz in self.dummyZONES:
                arr.append(self.arrival_dist[dz].rvs()[0].tolist())
                arr_count += sum(arr[i])
                i += 1
            tmpArrivalTime = {}
            for dz in self.dummyZONES:
                dzID = self.dummyZONES.index(dz)
                tmpArrivalTime[dz] = {}
                tmpSuccVessel, tmpFailVessel = self.getFailVessel(arr[dzID], self.HORIZON)
                count = -1
                tmpArrivalTime[dz] = {}
                tmpArrivalTime[dz]['succ'] = {}
                tmpArrivalTime[dz]['fail'] = {}
                for tt in range(self.HORIZON):
                    count += 1
                    if tmpSuccVessel[count] != 0:
                        tmpArrivalTime[dz]['succ'][tt] = tmpSuccVessel[count]
                        tmpArrivalTime[dz]['fail'][tt] = tmpFailVessel[count]
            self.arrivalTime = tmpArrivalTime
            iniCount = self.initialCount.rvs()[0]
            cT.nt_z = iniCount.copy()
            cT.nt_zt[:,0] = iniCount.copy()
            for dz in self.dummyZONES:
                dzID = self.dummyZONES.index(dz)
                nbr = nx.neighbors(self.zGraph, dz)
                if len(nbr) == 1:
                    cT.nt_zz[dz][nbr[0]][0] = sum(arr[dzID])
                    cT.nt_z[dz] = sum(arr[dzID])
                else:
                    print("Nbr of dummyZone > 1")
                    exit()
            self.total_vessels = int(iniCount.copy().sum() + arr_count)
        else:
            # -------- New Arrival Distribution
            id = random.choice(self.arr_seeds)
            self.arrivalTime = self.arrivalTimeDict[id]

            # -------- Initial Count
            nDummy = len(self.dummyZONES)
            for dz in self.dummyZONES:
                nbr = nx.neighbors(self.zGraph, dz)
                if len(nbr) == 1:
                    if self.total_vessels % nDummy > 0 and dz == 0:
                        pop = int(self.total_vessels / nDummy) + int(self.total_vessels % nDummy)
                    else:
                        pop = int(self.total_vessels / nDummy)
                    cT.nt_zz[dz][nbr[0]][0] = pop
                    cT.nt_z[dz] = pop
                else:
                    print ("Nbr of dummyZone > 1")
                    exit()
        return cT

    def step(self, t, cT, xi, pi):

        # self.beta_all[t] = beta
        self.cT = cT
        self.t = t
        '''Required only during rendering'''
        # self.zCount = updateZoneCount(self.t, self.cT, self.instance.zGraph)
        # self.reward.append(self.rt)  # Required for rendering
        # ---------- Rewards ----------- #
        self.rt_z, res, delay = self._atomicReward(self.cT.nt_z)
        self.rt = np.sum(self.rt_z * self.cT.nt_z)
        res = np.sum(res * self.cT.nt_z)
        delay = np.sum(delay * self.cT.nt_z)

        # Normalize the reward
        self.rt_z = self.rt_z / self.maxReward
        # -------------------------------------------- #
        # self.cT, countHash = self._sample(t, self.cT, xi, pi)
        # return self.rt, self.rt_z, self.cT, countHash
        self.cT = self._sample(t, self.cT, xi, pi)
        return self.rt, self.rt_z, self.cT, res, delay

    def _atomicReward(self, nt_z):

        rm = np.zeros((self.totalZONES))
        res = np.zeros((self.totalZONES))
        delay = np.zeros((self.totalZONES))
        for z in self.planningZONES:
            rm_z, res_z, delay_z = cost_t_nt_z(z, nt_z[z])
            rm[z] = -1 * rm_z
            res[z] = -1 * res_z
            delay[z] = -1 * delay_z
        return rm, res, delay

    def _sample(self, t, cT, xi_all, pi):

        cT.count_hash = []
        zGraph = self.zGraph
        # ------- Sample z' --------- #
        ztz = np.zeros((self.totalZONES, self.totalZONES))
        for z in self.planningZONES:
            pBucket = self.Pzz_mat[z]
            if sum(pBucket) > 0:
                ztz[z] = np.random.multinomial(cT.nt_zt[z][t], pBucket)

        # -------- Sample \nt(z,z',\omega)
        for z in self.planningZONES:
            for zp in nx.neighbors(zGraph, z):
                pBucket = pi[z][zp]
                cT.nt_z_o[z][zp] = np.random.multinomial(ztz[z][zp], pBucket)

        # ------- Sample \tau' ------ #
        notReached = np.zeros(self.totalZONES)
        vizDataTmp = []
        for z in self.planningZONES:
            for zp in nx.neighbors(zGraph, z):
                for o in range(self.num_options):
                    tMin = self.T_min_max[z][zp][0]
                    tMax = self.T_min_max[z][zp][1]
                    n = tMax - tMin
                    xi = xi_all[z][zp][o]
                    tmpSample = np.random.binomial(n, xi, int(cT.nt_z_o[z][zp][o]))
                    countBuckets = []
                    for c in range(n + 1):
                        countBuckets.append(np.count_nonzero(tmpSample == c))
                    if len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1]) < len(countBuckets):
                        len_nt = len(cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1])
                        len_pB = len(countBuckets)
                        cT.nt_z_o_tau[z][zp][o][t + tMin:t + tMax + 1] = countBuckets[0:len_nt]
                        cT.nt_zt_zt[z][t][zp][t + tMin:t + tMax + 1] += countBuckets[0:len_nt]
                        if sum(countBuckets[len_nt:len_pB]) > 0:
                            notReached[z] += sum(countBuckets[len_nt:len_pB])
                    else:
                        cT.nt_z_o_tau[z][zp][o][t+tMin:t+tMax+1] = countBuckets
                        cT.nt_zt_zt[z][t][zp][t+tMin:t+tMax+1] += countBuckets
                    if ztz[z][zp] > 0:
                        t1 = np.where(cT.nt_zt_zt[z][t][zp]> 0)[0]
                        for tp in t1:
                            trv_time = tp - t
                            count = cT.nt_zt_zt[z][t][zp][tp]
                            if count == 0:
                                print ("count")
                                exit()
                            for v in range(int(count)):
                                vizDataTmp.append((z, trv_time, tp))

        if len(vizDataTmp) > 0:
            self.vizData[t] = vizDataTmp

        # -------- Update Count Tables for next system time ------- #
        for z in range(self.totalZONES):
            if z not in self.dummyZONES:
                # ------- cT.nt_zt (Newly Arrived Agents) ------- #
                cT.nt_zt[z][t+1] = sum(cT.nt_zz[:,z,t+1]) + sum(cT.nt_zt_zt[:,t,z,t+1]) + notReached[z]

                # ------- cT.nt_zz (Transiting Agents)------- #
                for zp in range(self.totalZONES):
                    cT.nt_zz[z][zp][t+2:] = cT.nt_zz[z][zp][t+2:] + cT.nt_zt_zt[z][t][zp][t+2:]

                # ------ cT.nt_z ( Total Agents present in zone z)------ #
                cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(self.totalZONES)]))

                # ------ cT.ntz_zt (used in fictitious)----- #
                cT.ntz_zt[z] = cT.nt_zt_zt[z][t]

        # ------- Update Dummy Zone Count ------ #
        for dz in self.dummyZONES:
            for tp in self.arrivalTime[dz]['succ']:
                if t + 1 == tp:
                    succVessel = self.arrivalTime[dz]['succ'][tp]
                    failVessel = self.arrivalTime[dz]['fail'][tp]
                    pz = nx.neighbors(zGraph, dz)[0]
                    cT.nt_zt[pz][t+1] += succVessel
                    cT.nt_z[pz] += succVessel
                    cT.nt_z[dz] = failVessel

        # ------- Update Terminal Zone Count -------- #
        for z in self.termZONES:
            cT.nt_zt[z][t + 1] += cT.nt_zt[z][t]
            cT.nt_z[z] = cT.nt_zt[z][t+1] + sum(map(lambda zpp : sum(cT.nt_zz[z][zpp][t+2:]), [i for i in range(self.totalZONES)]))

        assert sum(cT.nt_z) == self.total_vessels, "Count Error : "+str(cT.nt_z)+" | "+str(self.total_vessels)+" != "+str(cT.nt_z.sum())

        return cT

def main():
    print ("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
