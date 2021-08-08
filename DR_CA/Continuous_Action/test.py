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
import math
import pandas as pd
# =============================== Variables ================================== #

# ============================================================================ #


def tplot():

    path = "/home/james/Codes/mpa/james4/cont_act/log/diff_mid_hyp_cont_map9"
    t1 = np.load(path+"/dr_cont.npy")
    t2 = t1.mean(axis=0)
    plt.plot(t2, label="cont")


    path = "/home/james/Codes/mpa/james4/cont_act/log/diff_max_hyp_map9"
    t1 = np.load(path+"/dr_disc.npy")
    t2 = t1.mean(axis=0)
    plt.plot(t2, label="disc")

    plt.legend()
    plt.show()
    exit()
    # set_trace()

def main():
    exp_data = pd.read_table('progress.txt')
    set_trace()

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