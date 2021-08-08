"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 11 Apr 2020
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.grid()
import seaborn
# plt.ion()
sys.dont_write_bytecode = True
import os
from pprint import pprint
import time
from ipdb import set_trace
import random
import numpy as np
import matplotlib.pyplot as plt
from auxLib3 import file2list, adjBar, plotMovingAverage, loadDataStr
import numpy as np
import random
# =============================== Variables ================================== #

# ============================================================================ #

def arr1(arr, h, max_ac):

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


def test():




    arr = [2, 4, 6]
    # arr = [4, 8]
    map_id = "map8"
    max_ac = 200
    horizon = 400

    pat1 = arr1(arr, horizon, max_ac/2)

    print(pat1)
    x = [_ for _ in range(horizon)]
    y = np.zeros(horizon)
    y[pat1] = 1

    en = 400
    x = x[0:en]
    y = y[0:en]

    # plt.subplot(911)
    # plt.plot(x, y, label=str(arr))
    # plt.legend()

    # arr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]#, 0.9, 1]
    arr_list = [0.9, 1]

    for i in range(len(arr_list)):
        arr_rate = arr_list[i]
        pat2 = loadDataStr("./routes_james/" + map_id+ "/" + "arr_rate_arrival_" + str(horizon) + "_" + str(
            arr_rate) + "_" + str(max_ac))
        y = np.zeros(horizon)
        t1 = pat2[0]
        y[t1] = 1
        y = y[0:en]
        # t1 = int(str("91") + str(i + 2))
        t1 = int(str("21")+str(i+1))
        plt.subplot(t1)
        plt.plot(x, y, label=str(arr_rate))
        plt.legend()


    plt.show()

    exit()



    # print(t1)

    set_trace()

def main():

    test()


# =============================================================================== #

if __name__ == '__main__':
    main()


'''
def plot_im_rw():

    pro_folder = os.getcwd()
    diff = np.load(pro_folder+"/log/diff_rw_v2_grad_sep_map4/im_rw.npy")

    # figcount = 0
    for e in range(diff.shape[0]):
        sbplt = int(str(4) + str(2) + str(e+1))
        plt.subplot(sbplt)
        plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
        plt.xlabel("Time Steps")
        plt.title(str(e))
        plt.plot(diff[e], label="Diff")
        plt.legend()

    plt.show()
'''