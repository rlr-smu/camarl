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
import os
from pprint import pprint
import time
import auxLib as ax
import pdb
import rlcompleter
from parameters import SEED, TOTAL_VESSEL, capRatio, BUFFER_MIN, BUFFER_MAX, LOAD_MODEL, T_MIN, T_MAX_MIN,T_MAX_MAX,  MODES, ARR_HORIZON, HORIZON, vSize, BATCH_SIZE, NUM_OPTIONS, HORIZON, INSTANCE, DENSITY, BUFFER_CAP, NUM_ARR_DIST, RANDOM_CAP, RANDOM_CAP_MIN, RANDOM_CAP_MAX, T_MIN_PERCENTILE, T_MAX_PERCENTILE, REAL_HORIZON, DPATH, RANDOM_TRAVEL, NUM_ACTIONS
import numpy as np
import networkx as nx
from auxLib import average, loadDataStr, dumpDataStr, file2list
import math
import matplotlib.pyplot as plt
import scipy as sp

np.random.seed(SEED)

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #

class env_data:

    def __init__(self, mapName=None):

        # ----- Discrete Actions
        self.num_actions = NUM_ACTIONS
        self.action_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        

        self.batch_size = BATCH_SIZE
        self.num_options = NUM_OPTIONS
        self.map_id = mapName
        self.seed = SEED
        self.total_vessels = TOTAL_VESSEL

        if "_" in mapName:
            # Lucas's Map
            mp = lucasMap(mapID=mapName)

        elif "x" in mapName:
            # Grid Map
            mp = gridMap(mapName)

        elif "real" in mapName:

            self.dpath = DPATH
            inst = loadDataStr(self.dpath+"/instance_H"+str(REAL_HORIZON))
            self.totalZONES = inst['totalZONES']
            self.termZONES = inst['termZONES']
            self.dummyZONES = inst['dummyZONES']
            self.planningZONES = inst['planningZONES']
            self.totalResource = inst['totalResource']
            self.HORIZON = inst['HORIZON']
            self.bufferZONES = inst['bufferZONES']
            self.dummyNbrs = inst['dummyNbrs']
            self.Pzz_mat = inst['Pzz_mat']
            self.rCon = inst['rCon']
            self.termNbrs = inst['termNbrs']
            self.planningTermZones = inst['planningTermZones']
            self.real_horizon = REAL_HORIZON

            # --------- zGraph
            # self.zGraph = inst['zGraph']
            self.zGraph = nx.DiGraph()
            with open(self.dpath+"/zGraph.txt") as f:
                for line in f:
                    t1 = line.split(",")
                    self.zGraph.add_edge(int(t1[0].strip()), int(t1[1].strip()))

            # --------- rCap
            self.rCap = inst['rCap']
            self.rCap = map(lambda v: int(v * capRatio), self.rCap)

            # --------- T_min_max
            self.T_min_max = np.array(
                [[(-1, -1) for _ in range(self.totalZONES)] for _ in range(self.totalZONES)])

            if T_MAX_PERCENTILE == -1:
                print("Error: T_min_max == (-1, -1)")
                exit()

            for z in self.planningZONES:
                for zp in nx.neighbors(self.zGraph, z):
                    ky = str(T_MIN_PERCENTILE) + "_" + str(T_MAX_PERCENTILE)
                    if (z, zp) in inst['T_min_max'][ky]:
                        if inst['T_min_max'][ky][(z, zp)][0] == -1 or inst['T_min_max'][ky][(z,zp)][1] == -1:
                            self.T_min_max[z][zp] = inst['T_min_max']['fix'][(z,zp)]
                        else:
                            self.T_min_max[z][zp] = inst['T_min_max'][ky][(z,zp)]
                    elif (z, zp) in inst['T_min_max']['fix']:
                        self.T_min_max[z][zp] = inst['T_min_max']['fix'][(z, zp)]

            for z in self.dummyNbrs:
                for zp in nx.neighbors(self.zGraph, z):
                    self.T_min_max[z][zp][0] = BUFFER_MIN
                    self.T_min_max[z][zp][1] = BUFFER_MAX

            flag = False
            for z in self.planningZONES:
                for zp in nx.neighbors(self.zGraph, z):
                    if self.T_min_max[z][zp][0] == -1 or self.T_min_max[z][zp][1] == -1:
                        print("Error : T_min_max = (-1, -1) for zone-pair : " + "(" + str(z) + "," + str(zp) + ")")
                        flag = True
            if flag:
                exit()

            # --------- Arrival
            self.arrival_dist = inst['arrival']

            # -------- Initial Count
            self.initialCount = inst['initCount']
            # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
            # pdb.set_trace()

            return

        else:
            print("Error: Map not defined !")
            exit()

        self.total_vessels = TOTAL_VESSEL
        self.HORIZON = HORIZON
        self.num_arr_dist = NUM_ARR_DIST
        self.totalZONES = mp.totalZONES
        self.dummyZONES = mp.dummyZONES
        self.termZONES = mp.termZONES
        self.zGraph = mp.G
        self.rCap = mp.rCap
        self.arr_seeds = mp.arr_seeds
        self.arrivalTimeDict = mp.arrivalTimeDict
        self.T_min_max = mp.T_min_max
        self.dummyNbrs = mp.dummyNbrs
        self.totalResource = self.totalZONES
        self.planningZONES = list(set([_ for _ in range(self.totalZONES)]) - set(self.dummyZONES) - set(self.termZONES))
        self.planningTermZones = []
        self.planningTermZones.extend(self.planningZONES)
        self.planningTermZones.extend(self.termZONES)

        # ----------- Pzz ----------- #
        self.Pzz_mat = np.ones((self.totalZONES, self.totalZONES))

        for z in range(self.totalZONES):
            for z_next in range(self.totalZONES):
                if z_next in self.dummyZONES:
                    self.Pzz_mat[z][z_next] = 0 #TINY
                if z_next not in nx.neighbors(self.zGraph, z):
                    self.Pzz_mat[z][z_next] = 0 #TINY

        # ---- Normalize ---- #
        for z in range(self.totalZONES):
            for z_next in range(self.totalZONES):
                if np.sum(self.Pzz_mat[z]) > 0:
                    self.Pzz_mat[z] = self.Pzz_mat[z]/np.sum(self.Pzz_mat[z])

        for z in self.planningZONES:
            assert np.sum(self.Pzz_mat[z]) == 1, "Error in Pzz_mat"

        # --------- Resource Consumption
        self.rCon = np.zeros((self.totalZONES, self.totalZONES))
        for z in self.planningZONES:
            for r in range(self.totalResource):
                if z == r:
                    self.rCon[z][r] = vSize

        # ------ termNbrs
        termNbrs = []
        for z in self.planningZONES:
            z_nbrs = nx.neighbors(self.zGraph, z)
            for zt in self.termZONES:
                if zt in z_nbrs:
                    termNbrs.append(z)

class realData:

    def __init__(self, mapName=None):


        print ("from real data class")

class lucasMap:

    def __init__(self, mapID=None):


        self.G, self.totalZONES, self.termZONES, self.dummyZONES = self.loadMap(mapID=mapID, instance=INSTANCE)
        np.random.seed(SEED)

        # ------ dummyNbrs
        self.dummyNbrs = []
        for zd in self.dummyZONES:
            z_nbr = nx.neighbors(self.G, zd)
            self.dummyNbrs.extend(z_nbr)
        self.dummyNbrs = list(set(self.dummyNbrs))

        # ------ T_Min_Max
        if RANDOM_TRAVEL:
            self.T_min_max = np.array([[(-1, -1) for _ in range(self.totalZONES)] for _ in range(self.totalZONES)])
            for z in range(self.totalZONES):
                for zp in range(self.totalZONES):
                    if z != zp:
                        self.T_min_max[z][zp][0] = T_MIN
                        self.T_min_max[z][zp][1] = np.random.randint(T_MAX_MIN, T_MAX_MAX)
        else:
            self.T_min_max = np.array([[(T_MIN, T_MAX_MAX) for _ in range(self.totalZONES)] for _ in range(self.totalZONES)])
        for z in self.dummyNbrs:
            for zp in nx.neighbors(self.G, z):
                self.T_min_max[z][zp][0] = BUFFER_MIN
                if LOAD_MODEL:
                    self.T_min_max[z][zp][1] = BUFFER_MAX
                else:
                    self.T_min_max[z][zp][1] = BUFFER_MAX

        # ------ Arrival Time
        if NUM_ARR_DIST == 1:
            self.arr_seeds = [SEED]
        else:
            self.arr_seeds = np.random.choice(100, NUM_ARR_DIST, replace=False)

        self.arrivalTimeDict = {}
        for i in range(NUM_ARR_DIST):
            tmpArrivalTime = {}
            arr = mmArrivalMulti(len(self.dummyZONES), MODES, ARR_HORIZON, HORIZON, TOTAL_VESSEL, self.arr_seeds[i])
            mmArrival = arr.getArrival()
            for dz in self.dummyZONES:
                dzID = self.dummyZONES.index(dz)
                tmpArrivalTime[dz] = {}
                tmpSuccVessel, tmpFailVessel = getFailVessel(mmArrival[dzID], HORIZON)
                count = -1
                tmpArrivalTime[dz] = {}
                tmpArrivalTime[dz]['succ'] = {}
                tmpArrivalTime[dz]['fail'] = {}
                for tt in range(HORIZON):
                    count += 1
                    if tmpSuccVessel[count] != 0:
                        tmpArrivalTime[dz]['succ'][tt] = tmpSuccVessel[count]
                        tmpArrivalTime[dz]['fail'][tt] = tmpFailVessel[count]
            self.arrivalTimeDict[self.arr_seeds[i]] = tmpArrivalTime

        # ------ Resource Capacity
        if RANDOM_CAP:
            self.rCap = np.random.randint(RANDOM_CAP_MIN, int(RANDOM_CAP_MAX*capRatio), size=self.totalZONES)
            self.rCap = list(self.rCap)
        else:
            self.rCap = map(lambda v: int(RANDOM_CAP_MAX), [i for i in range(self.totalZONES)])
            self.rCap = map(lambda v: int(v * capRatio), self.rCap)
        for z in self.dummyNbrs:
            self.rCap[z] = BUFFER_CAP

    def loadMap(self, mapID=None, instance=None):

        if os.path.exists("./synData/lucas/"+mapID+"_"+str(instance)):
            G = loadDataStr("./synData/lucas/"+mapID+"_"+str(instance)+"/G")
            totalZONES = loadDataStr("./synData/lucas/"+mapID+"_"+str(instance)+"/totalZ")
            termZONES = loadDataStr("./synData/lucas/"+mapID+"_"+str(instance)+"/termZ")
            dummyZONES = loadDataStr("./synData/lucas/"+mapID+"_"+str(instance)+"/dummyZ")
            return G, totalZONES, termZONES, dummyZONES
        else:
            print("Error : Map not created!")
            exit()

class gridMap:

    def __init__(self, mapID):

        # --------- Graph
        fname = "./synData/grid/"+mapID+"/dG.txt"
        if os.path.exists(fname):
            f = file2list(fname)
            # ------ Edges
            dG = nx.DiGraph()
            for line in f:
                if "edges" in line:
                    e = line.split(":")[1].split(",")
                    dG.add_edge(int(e[0]), int(e[1]))
            # ----- termZONES
            termZONES = []
            for line in f:
                if "termZONES" in line:
                    e = line.split(":")[1].split(",")
                    for n in e:
                        termZONES.append(int(n))
            # ----- dummyZONES
            dummyZONES = []
            for line in f:
                if "dummyZONES" in line:
                    e = line.split(":")[1].split(",")
                    for n in e:
                        dummyZONES.append(int(n))
            totalZONES = len(dG.nodes())
            self.G = dG
            self.totalZONES = totalZONES
            self.termZONES = termZONES
            self.dummyZONES = dummyZONES
        else:
            print ("Error: Grid Graph Not Found !")
            exit()

        # ------ dummyNbrs
        self.dummyNbrs = []
        for zd in self.dummyZONES:
            z_nbr = nx.neighbors(self.G, zd)
            self.dummyNbrs.extend(z_nbr)
        self.dummyNbrs = list(set(self.dummyNbrs))

        # ------ T_Min_Max
        self.T_min_max = np.array([[(-1, -1) for _ in range(self.totalZONES)] for _ in range(self.totalZONES)])
        for z in range(self.totalZONES):
            for zp in range(self.totalZONES):
                if z != zp:
                    self.T_min_max[z][zp][0] = T_MIN
                    self.T_min_max[z][zp][1] = np.random.randint(T_MAX_MIN, T_MAX_MAX)
        for z in self.dummyNbrs:
            for zp in nx.neighbors(self.G, z):
                self.T_min_max[z][zp][0] = BUFFER_MIN
                if LOAD_MODEL:
                    self.T_min_max[z][zp][1] = BUFFER_MAX
                else:
                    self.T_min_max[z][zp][1] = BUFFER_MAX

        # ------ Arrival Time
        if NUM_ARR_DIST == 1:
            self.arr_seeds = [SEED]
        else:
            self.arr_seeds = np.random.choice(100, NUM_ARR_DIST, replace=False)
        self.arrivalTimeDict = {}
        for i in range(NUM_ARR_DIST):
            tmpArrivalTime = {}
            arr = mmArrivalMulti(len(self.dummyZONES), MODES, ARR_HORIZON, HORIZON, TOTAL_VESSEL, self.arr_seeds[i])
            mmArrival = arr.getArrival()
            for dz in self.dummyZONES:
                dzID = self.dummyZONES.index(dz)
                tmpArrivalTime[dz] = {}
                tmpSuccVessel, tmpFailVessel = getFailVessel(mmArrival[dzID])
                count = -1
                tmpArrivalTime[dz] = {}
                tmpArrivalTime[dz]['succ'] = {}
                tmpArrivalTime[dz]['fail'] = {}
                for tt in range(HORIZON):
                    count += 1
                    if tmpSuccVessel[count] != 0:
                        tmpArrivalTime[dz]['succ'][tt] = tmpSuccVessel[count]
                        tmpArrivalTime[dz]['fail'][tt] = tmpFailVessel[count]
            self.arrivalTimeDict[self.arr_seeds[i]] = tmpArrivalTime

        # ------ Resource Capacity
        if RANDOM_CAP:
            self.rCap = np.random.randint(RANDOM_CAP_MIN, int(RANDOM_CAP_MAX*capRatio), size=self.totalZONES)
            self.rCap = list(self.rCap)
        else:
            self.rCap = map(lambda v: int(RANDOM_CAP_MAX), [i for i in range(self.totalZONES)])
            self.rCap = map(lambda v: int(v * capRatio), self.rCap)
        for z in self.dummyNbrs:
            self.rCap[z] = BUFFER_CAP

    def createGridGraph(self, mapID):
        pass

class mmArrivalMulti:

    def __init__(self, source, modes, arrHorizon, horizon, population, seed):

        # assert(source%2 == 0), "Source not even number : "+str(source)
        self.SOURCE = source
        self.MODES = modes
        # self.HPARAM = hParam
        self.ARR_HORIZON = arrHorizon
        self.TOTAL_HORIZON = horizon
        self.TOTAL_AGENTS = population
        self.SEED = seed
        self.Arrival = self.getArrival()

    def getArrival(self):
        tmp = []
        for s in range(self.SOURCE):
            if self.TOTAL_AGENTS % self.SOURCE > 0 and s == 0:
                xtra = int(self.TOTAL_AGENTS % self.SOURCE)
                pop = int(self.TOTAL_AGENTS) / self.SOURCE + xtra
            else:
                pop = int(self.TOTAL_AGENTS) / self.SOURCE
            # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
            # pdb.set_trace()
            uni = mmArrival(self.MODES, self.ARR_HORIZON, self.TOTAL_HORIZON, pop, self.SEED+s)
            arr = uni.getArrival()
            tmp.append(arr)
        return tmp

class mmArrival:
    def __init__(self, modes, arrHorizon, horizon, population, seed):
        """
        :param hParam: (0, 1) fraction of total horizon
        :param h: HORIZON
        :param p: Total Population
        :param mode: No. of Modes
        """
        self.SEED = seed
        np.random.seed(self.SEED)
        sp.random.seed(self.SEED)
        self.SHAPE = (0.4, 0.6)
        self.TOTAL_HORIZON = horizon
        self.ARR_HORIZON = arrHorizon
        # self.HORIZON = int(self.TOTAL_HORIZON * self.HPARAM)
        # self.HORIZON = int(math.ceil(self.TOTAL_HORIZON * self.HPARAM))
        self.HORIZON = self.ARR_HORIZON
        self.TOTAL_AGENTS = population
        self.MODES = modes
        self.ARRIVAL = self.getArrival()
        # self.plotArrival()

    def getArrival(self):
        tmp = self.getMultiModal()
        tmp.extend([ 0 for _ in range(self.TOTAL_HORIZON - self.HORIZON)])
        if tmp[0] > 0:
            tmp[1] += tmp[0]
            tmp[0] = 0
        self.ARRIVAL = tmp
        return tmp

    def getMultiModal(self):

        modData = self.getModeHorizonPop(self.MODES, self.HORIZON, self.TOTAL_AGENTS)
        tmp = []
        for mID in modData:
            h = modData[mID]['h']
            m = modData[mID]['m']
            uni = self.getUnimodal(h, m)
            tmp.extend(uni)
        assert (sum(tmp) == self.TOTAL_AGENTS), "Population not matched "+str(self.TOTAL_AGENTS)+" != "+str(sum(tmp))
        assert (len(tmp) == self.HORIZON), "Horizon not matched "+str(self.HORIZON)+" != "+str(len(tmp))
        return tmp

    def getUnimodal(self, h, m):
        h -= 1
        h = int(h)
        p = np.random.uniform(self.SHAPE[0], self.SHAPE[1])
        s = sp.stats.binom(h, p)
        prob = s.pmf([_ for _ in range(h + 1)])
        tmp = np.random.multinomial(m, prob)
        return tmp

    def getModeHorizonPop(self, modes, H, M):
        tmp = {}
        r = H % modes
        q = H / modes
        rM = M % modes
        qM = M / modes
        for m in range(1, modes+1):
            tmp[m] = {}
            tmp[m]['h'] = q
            tmp[m]['m'] = qM
        tmp[modes]['h'] += r
        tmp[modes]['m'] += rM
        tmp2 = []
        for m in range(1, modes+1):
            tmp2.append(tmp[m]['m'])
        assert(sum(tmp2) == M), "Population not matched "+str(M)+" != "+str(sum(tmp2))
        return tmp

    def plotArrival(self):
        y = self.ARRIVAL
        x = [_ for _ in range(len(y))]
        plt.xlabel("Horizon")
        plt.ylabel("Agent Count")
        plt.bar(x, y)
        plt.grid()
        # plt.show()
        # plt.savefig("Arrival.png")
        plt.savefig("Arrival_"+str(self.HPARAM)+"_"+str(self.MODES)+"_"+str(self.TOTAL_AGENTS)+"_"+str(self.TOTAL_HORIZON)+".png")
        # exit()

def getFailVessel(tmpSuccVessel, HORIZON):
    totalHorizon = HORIZON
    tmpSuccVessel.append(0)
    tmpPop = sum(tmpSuccVessel)
    tmpFailVessel = [0]
    for tt in range(1, totalHorizon + 1):
        tmpFail = tmpPop - sum(tmpSuccVessel[1:tt+1])
        tmpFailVessel.append(tmpFail)
    return tmpSuccVessel, tmpFailVessel

class mapStats:

    def __init__(self, data=None):



        pass

# ============================================================================ #

def main():

    # mapList = ["5_3", "6_3", "7_3", "8_3", "9_3", "10_3"]
    # mapList = ["5_3", "6_3", "7_3", "8_3", "9_3"]
    mapList = ["5_2", "10_2", "15_2", "20_2"]

    for i in range(len(mapList)):
        mp = mapList[i]
        data = env_data(mapName=mp)
        dzList = data.dummyZONES
        termZones = data.termZONES
        G = data.zGraph
        e = 0
        for (a, b) in G.edges():
            if a not in dzList and b not in termZones:
                e += 1
        print (mp, len(data.planningZONES), len(G.nodes())-1)


        # print mp, len(G.edges())
        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        #pdb.set_trace()

    #
    # exit()
    # data = env_data(mapName="2x2")
    # data = env_data(mapName="real")
    pass


    # dt = loadDataStr("synData/real_map/2017-01-12")
    # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
    # pdb.set_trace()



# =============================================================================== #

if __name__ == '__main__':
    main()
