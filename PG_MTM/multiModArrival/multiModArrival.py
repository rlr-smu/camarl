"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 08 Mar 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print "# ============================ START ============================ #"
# ================================ Imports ================================ #
import sys
import os
import platform
from pprint import pprint
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn
from pprint import pprint
import math

# ============================================================================ #
# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location

# -------------------------------------------------------------- #

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
            pop = int(self.TOTAL_AGENTS)/self.SOURCE
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
        assert (sum(tmp) == self.TOTAL_AGENTS), "Population not matched "+str(self.TOTAL_AGENTS)+" <> "+str(sum(tmp))
        assert (len(tmp) == self.HORIZON), "Horizon not matched "+str(self.HORIZON)+" <> "+str(len(tmp))
        return tmp

    def getUnimodal(self, h, m):
        h -= 1
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
        assert(sum(tmp2) == M), "Population not matched "+str(M)+" <> "+str(sum(tmp2))
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
        plt.savefig(os.getcwd()+"/multiModArrival/"+"Arrival_"+str(self.HPARAM)+"_"+str(self.MODES)+"_"+str(self.TOTAL_AGENTS)+"_"+str(self.TOTAL_HORIZON)+".png")
        # exit()


def main():

    # arr = mmArrival(3, 0.6, 100, 100, 1)
    # Arrival = arr.getArrival()
    # print Arrival, sum(Arrival)

    arr = mmArrivalMulti(2, 3, 0.4, 100, 100, 1)
    Arrival = arr.getArrival()
    print Arrival[0], sum(Arrival[0])
    print Arrival[1], sum(Arrival[1])


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
    