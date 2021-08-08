"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 22 Mar 2020
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
# from plugins.utils import auxLib
# from plugins.utils import plotMovingAverage
import matplotlib.pyplot as plt
import numpy as np
from auxLib3 import file2list, adjBar


# =============================== Variables ================================== #
# ax = auxLib()
global plt

# ============================================================================ #

def average_metric(fpath, sample_size):
    '''
   Total Reward : 69.65
   Avg. Travel Time : 188.0,13.601
   Avg. Conflicts : 0.643,1.066
   Max  Conflicts : 4.0
    :return:
    '''

    # fpath = "log/test_1/log_single.txt"
    fl = file2list(fpath)
    y_rw = []
    y_av_confl = []
    y_mx_confl = []
    y_tr = []
    for i in range(len(fl)):
        line = fl[i]
        if "Total Reward" in line:
            t1 = float(line.split(" : ")[1])
            y_rw.append(t1)
        if "Avg. Travel Time" in line:
            t1 = line.split(" : ")[1]
            t1 = float(t1.split(",")[0])
            y_tr.append(t1)
        if "Avg. Conflicts" in line:
            t1 = line.split(" : ")[1]
            t1 = float(t1.split(",")[0])
            y_av_confl.append(t1)
        if "Max  Conflicts" in line:
            t1 = float(line.split(" : ")[1])
            y_mx_confl.append(t1)


    y_rw = y_rw[-sample_size:]
    y_av_confl = y_av_confl[-sample_size:]
    y_mx_confl = y_mx_confl[-sample_size:]
    y_tr = y_tr[-sample_size:]


    tmp_data = {}

    tmp_data["avg_conf"] = (np.mean(y_av_confl), np.std(y_av_confl))
    tmp_data["avg_travel"] = (np.mean(y_tr), np.std(y_tr))
    tmp_data["reward"] = (np.mean(y_rw), np.std(y_rw))
    tmp_data["max_conf"] = (np.mean(y_mx_confl), np.std(y_mx_confl))

    return tmp_data

def best_metric(fpath):
    '''
   Total Reward : 69.65
   Avg. Travel Time : 188.0,13.601
   Avg. Conflicts : 0.643,1.066
   Max  Conflicts : 4.0
    :return:
    '''

    fl = file2list(fpath)
    y_rw = []
    y_av_confl = []
    y_av_confl_std = []
    y_mx_confl = []
    y_tr = []
    y_tr_std = []
    for i in range(len(fl)):
        line = fl[i]
        if "Total Reward" in line:
            t1 = float(line.split(" : ")[1])
            y_rw.append(t1)
        if "Avg. Travel Time" in line:
            t1 = line.split(" : ")[1]
            t2 = float(t1.split(",")[0])
            t3 = float(t1.split(",")[1])
            y_tr.append(t2)
            y_tr_std.append(t3)
        if "Avg. Conflicts" in line:
            t1 = line.split(" : ")[1]
            t2 = float(t1.split(",")[0])
            t3 = float(t1.split(",")[1])
            y_av_confl.append(t2)
            y_av_confl_std.append(t3)
        if "Max  Conflicts" in line:
            t1 = float(line.split(" : ")[1])
            y_mx_confl.append(t1)

    mx = max(y_rw)
    mx_ind = y_rw.index(mx)
    print("Reward", y_rw[mx_ind])
    print("Avg. Confl", y_av_confl[mx_ind], y_av_confl_std[mx_ind])
    print("Max Confl", y_mx_confl[mx_ind])
    print("Avg. Travel", y_tr[mx_ind], y_tr_std[mx_ind])

def main():

    ppath = "/home/james/Dropbox/Buffer/ExpResults/Project3/Results/April_31/Exp2/map4/compare"

    agents = {}

    agents['baseline'] = {}
    agents['baseline']['path'] = ppath+"/baseline/log_inf_ppo.txt"

    agents['count_diff_rw'] = {}
    agents['diff_rw']['path'] = ppath+"/count_diff_rw/log_inf.txt"

    agents['count_global'] = {}
    agents['count_global']['path'] = ppath+"/count_global/log_inf.txt"

    # agents['count_single'] = {}
    # agents['count_single']['path'] = "log/count_single/log_inf.txt"
    # agents['count_multi'] = {}
    # agents['count_multi']['path'] = "log/count_multi/log_inf.txt"


    for ag in agents:
        fp = agents[ag]['path']
        tmp_data = average_metric(fp, 500)
        agents[ag]['data'] = tmp_data


    # Reward
    bar_label = []
    y = []
    y_err = []
    for ag in agents:
        y.append([agents[ag]['data']['reward'][0]])
        y_err.append([agents[ag]['data']['reward'][1]])
        bar_label.append(ag)
    x = ["Reward"]
    pl = adjBar(x=x, barWidth=0.5, fsize='x-large', ylabel="Total Reward")
    pl.save(y=y, y_err=y_err, bar_label=bar_label, fname=ppath+"/reward.png")

    # Conflicts
    bar_label = []
    y = []
    y_err = []
    for ag in agents:
        y.append([agents[ag]['data']['avg_conf'][0], agents[ag]['data']['max_conf'][0]])
        y_err.append([agents[ag]['data']['avg_conf'][1], agents[ag]['data']['max_conf'][1]])
        bar_label.append(ag)
    x = ["avg_conflict", "max_conflict"]
    pl = adjBar(x=x, barWidth=0.1, legend_loc="upper center", fsize='x-large', ylabel="Conflicts")
    pl.save(y=y, y_err=y_err, bar_label=bar_label, fname=ppath+"/conflicts.png")


    # Travel Time
    bar_label = []
    y = []
    y_err = []
    for ag in agents:
        y.append([agents[ag]['data']['avg_travel'][0]])
        y_err.append([agents[ag]['data']['avg_travel'][1]])
        bar_label.append(ag)
    x = ["avg_travel"]
    pl = adjBar(x=x, barWidth=0.1, legend_loc="upper center", fsize='x-large', ylabel="Travel Time")
    pl.save(y=y, y_err=y_err, bar_label=bar_label, fname=ppath+"/travel.png")


# =============================================================================== #

if __name__ == '__main__':
    main()

