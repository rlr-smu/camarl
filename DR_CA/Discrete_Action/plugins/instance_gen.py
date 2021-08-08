"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 17 Jul 2020
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
from ipdb import set_trace
import pdb
import rlcompleter
from parameters import MAP_ID, ARR_RATE_INTERVAL, SEED, ARR_RATE
import numpy as np
from auxLib3 import loadDataStr, dumpDataStr
# import random
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
import matplotlib.pyplot as plt
# =============================== Variables ================================== #

np.random.seed(69)

# ============================================================================ #

class syn_data:

    def __init__(self, horizon=None, max_ac=None, arr_rate=None):

        self.horizon = horizon
        self.max_ac = max_ac
        self.arr_rate = arr_rate

    def gen_arrival(self):
        np.random.seed(SEED)
        pro_folder = os.getcwd()
        self.directions = loadDataStr(pro_folder+"/routes_james/"+MAP_ID+"/"+MAP_ID+"_directions")
        arr_horizon = int(self.horizon)
        tmp_routes = list(self.directions.keys())
        self.arr_rate_arrival = {}
        for i in tmp_routes:
            self.arr_rate_arrival[i] = []
            nm_ac_rt = int(self.max_ac/len(tmp_routes))
            n_intv = int(self.arr_rate * ARR_RATE_INTERVAL)
            if n_intv >= nm_ac_rt:
                n_intv = nm_ac_rt
            h_intv = int(arr_horizon / ARR_RATE_INTERVAL)
            for h in range(h_intv):
                if nm_ac_rt > 0:
                    st = h + h*ARR_RATE_INTERVAL
                    if st > arr_horizon:
                        break
                    en = st + ARR_RATE_INTERVAL
                    if en > arr_horizon:
                        en = arr_horizon
                    if en - (st+1) < n_intv:
                        n_intv = en-(st+1)
                    t1 = np.random.choice(range(st+1, en), n_intv, replace=False)
                    t1 = np.sort(t1)
                    self.arr_rate_arrival[i].extend(list(t1))
                    nm_ac_rt = nm_ac_rt - n_intv
        dumpDataStr(pro_folder+"/routes_james/"+MAP_ID+"/"+"arr_rate_arrival_"+str(self.horizon)+"_"+str(self.arr_rate)+"_"+str(self.max_ac), self.arr_rate_arrival)

class arr_analysis:

    def __init__(self):

        arr = [4, 8]
        horizon = 400
        max_ac = 50
        arr_rate = 0.3

        pat1 = self.arr1(arr, horizon, max_ac / 2)
        print(pat1)
        exit()
        pat2 = loadDataStr("./routes_james/" + MAP_ID+ "/" + "arr_rate_arrival_" + str(horizon) + "_" + str(
            arr_rate) + "_" + str(max_ac))

        print(pat2[0])
        print(pat2[1])
        exit()
        x = [_ for _ in range(horizon)]
        y = np.zeros(horizon)
        y[pat1] = 1
        en = 400
        x = x[0:en]
        y = y[0:en]
        plt.plot(x, y, label=str(arr))
        plt.legend()
        plt.show()


    def arr1(self, arr, h, max_ac):

        arr_pat = []
        na = np.random.choice(arr)
        y = np.zeros(h)
        for t in range(h):
            if t == na:
                arr_pat.append(na)
                na = t + random.choice(arr)
                if len(arr_pat) == max_ac:
                    break
        return arr_pat

class gen_arrival:

    def __init__(self, max_ac=None):
        np.random.seed(69)
        self.max_ac = max_ac
        self.nex_arr_int = {}
        self.nex_arr_int[0.1] = [6, 8]
        self.nex_arr_int[0.2] = [4, 8]
        self.nex_arr_int[0.3] = [2, 6]
        self.nex_arr_int[0.4] = [2, 4]
        self.nex_arr_int[0.5] = [2]
        self.pro_folder = os.getcwd()
        self.directions = loadDataStr(self.pro_folder+"/routes_james/"+MAP_ID+"/"+MAP_ID+"_directions")
        for inst in range(1, 11):
            os.system("mkdir "+self.pro_folder+"/routes_james/"+MAP_ID+"/instance/"+str(inst))
            for arr in self.nex_arr_int:
                arrival = self.arrival(arr)
                dumpDataStr(
                    self.pro_folder + "/routes_james/" + MAP_ID + "/instance/"+str(inst)+"/arr_rate_arrival_"+ str(
                        arr) + "_" + str(self.max_ac), arrival)

    def arrival(self, arr_rate):

            tmp_routes = list(self.directions.keys())
            arr_rate_arrival = {}
            arr_int_list = self.nex_arr_int[arr_rate]
            for i in tmp_routes:
                arr_rate_arrival[i] = []
                nm_ac_rt = int(self.max_ac/len(tmp_routes))
                ac_t = np.random.choice(arr_int_list)
                nex_ac_t = ac_t
                arr_rate_arrival[i].append(nex_ac_t)
                nm_ac_rt -= 1
                for t in range(1, 1000):
                    if t == nex_ac_t:
                        ac_t = np.random.choice(arr_int_list)
                        nex_ac_t = t + ac_t
                        arr_rate_arrival[i].append(nex_ac_t)
                        nm_ac_rt -= 1
                        if nm_ac_rt == 0:
                            break
            return arr_rate_arrival

class gen_arrival_bl:

    def __init__(self):
        np.random.seed(69)
        self.pro_folder = os.getcwd()
        self.directions = loadDataStr(self.pro_folder+"/routes_james/"+MAP_ID+"/"+MAP_ID+"_directions")
        self.num_entry = len(self.directions.keys())

        self.intr_arr = {}
        self.intr_arr[0.1] = [2, 3]
        self.intr_arr[0.1] = [4, 5, 6]
        self.intr_arr[0.1] = [6, 7, 9]
        self.intr_arr[0.1] = [8, 10, 12]
        self.intr_arr[0.1] = [10, 12, 15]
        self.intr_arr[0.1] = [12, 15, 18]
        self.intr_arr[0.1] = [14, 17, 21]
        self.intr_arr[0.1] = [16, 20, 24]
        self.intr_arr[0.1] = [18, 22, 27]
        self.intr_arr[0.1] = [20, 25, 30]





        for inst in range(1, 6):
            os.system("mkdir "+self.pro_folder+"/routes_james/"+MAP_ID+"/instance/"+str(inst))
            for arr in self.nex_arr_int:
                arrival = self.arrival(arr)
                dumpDataStr(
                    self.pro_folder + "/routes_james/" + MAP_ID + "/instance/"+str(inst)+"/arr_rate_arrival_"+ str(
                        arr) + "_" + str(self.max_ac), arrival)
    def arrival(self):


        for t in range(2000):

            if t == 0:
                for did in range(self.num_entry):
                    rid = np.random.choice(self.directions[did])
                    next_ac_t = np.random.choice(self.interval)
                    self.new_arrival_t[did] += next_ac_t
            else:
                for d_id in t1:
                    r_id = np.random.choice(self.directions[d_id])
                    route_keeper = self.add_new_aircraft(r_id, route_keeper)
                    next_ac_t = np.random.choice(self.interval)

                    # set_trace()
                    self.new_arrival_t[d_id] += next_ac_t

def main():

    # pro_folder = os.getcwd()
    # arr = loadDataStr(pro_folder + "/routes_james/" + MAP_ID + "/instance/"+str(1)+"/arr_rate_arrival_"+ str(
    #                     0.3) + "_" + str(50))
    #
    # print(arr)
    #
    # arr = loadDataStr(pro_folder + "/routes_james/" + MAP_ID + "/instance/"+str(1)+"/arr_rate_arrival_"+ str(
    #                     0.4) + "_" + str(50))
    #
    # print(arr)
    # set_trace()

    M = [50, 60, 70, 80, 90, 100, 150, 200]
    for m in M:
        gen_arrival(max_ac=m)
    # arr_analysis()
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    