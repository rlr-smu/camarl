"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 06 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ secImports ================================ #

import sys
import os
import platform
from pprint import pprint
import time

# o = platform.system()
# if o == "Linux":
#     d = platform.dist()
#     if d[0] == "debian":
#         sys.path.append("/home/james/Codes/python")
#     if d[0] == "centos":
#         sys.path.append("/home/arambamjs.2016/projects")
#     if d[0] == "redhat":
#         sys.path.append("/home/arambamjs/projects")
# if o == "Darwin":
#     sys.path.append("/Users/james/Codes/python")
# import auxlib.auxLib as ax
import auxLib as ax
from auxLib import average
from map.map import MAP

# ================================ priImports ================================ #
import numpy as np
from data import totalZONES, dummyZONES, HORIZON, totalResource, rCap, rCon, planningZONES, termZONES, Pzz, totalVESSELS, T_min_max, zGraph, dummyNbrs
import networkx as nx
import tensorflow as tf
from parameters import wDelay, wResource, MAP_ID


# ============================================================================ #

# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location

# -------------------------------------------------------------- #

def getTravelTime(t, cT, zGraph):

    tTravel = {}
    for z in planningZONES:
        timeSum = 0
        for zp in nx.neighbors(zGraph, z):
            tMin = T_min_max[z][zp][0]
            tMax = T_min_max[z][zp][1]
            tp_space = [ _ for _ in range(t+tMin, t+tMax+1)]
            tp_space = filter(lambda tp2 : tp2 <= HORIZON, tp_space)
            tmp2 = map(lambda tp : (tp - t) * cT.nt_zt_zt[z][t][zp][tp], tp_space)
            timeSum += np.sum(tmp2)
        if np.sum(cT.nt_zt_zt[z][t]) > 0:
            finalSum = timeSum / np.sum(cT.nt_zt_zt[z][t])
        else:
            finalSum = timeSum
        tTravel[z] = finalSum
    return average(tTravel.values())

def getTravelDelay(t, cT, zGraph):

    tTravel = {}
    for z in planningZONES:
        timeSum = 0
        for zp in nx.neighbors(zGraph, z):
            tMin = T_min_max[z][zp][0]
            tMax = T_min_max[z][zp][1]
            tp_space = [_ for _ in range(t + tMin, t + tMax + 1)]
            tp_space = filter(lambda tp2: tp2 <= HORIZON, tp_space)
            tmp2 = map(lambda tp: (tp - t - tMin) * cT.nt_zt_zt[z][t][zp][tp], tp_space)
            timeSum += np.sum(tmp2)
        if np.sum(cT.nt_zt_zt[z][t]) > 0:
            finalSum = timeSum / np.sum(cT.nt_zt_zt[z][t])
        else:
            finalSum = timeSum
        tTravel[z] = finalSum
    return average(tTravel.values())

    '''
    tmpSum = 0
    for z in planningZONES:
        for zp in nx.neighbors(zGraph, z):
            tmpSum += reduce(lambda v1, v2: v1 + v2,
                             map(lambda tp: (float(cT.nt_zt_zt[z][t][zp][tp]) / totalVESSELS) * (tp - t),
                                 [_ for _ in range(t + 1, HORIZON + 1)]))
    return tmpSum
    '''

def getResVioNoPenalty(nt_z, zGraph):
    def count(z, nt_z):
        return reduce(lambda v1, v2: v1 + v2,
                      map(lambda r: np.maximum(rCon[z][r] * nt_z - rCap[r], 0), [r for r in range(totalResource)]))
    tmp = np.zeros((totalZONES))
    for z in planningZONES:
        if z not in dummyNbrs:
            tmp[z] = count(z, nt_z[z])
        else:
            tmp[z] = 0
    return tmp

def getVioCount(nt_z, zGraph):
    def count(z, nt_z):
        return reduce(lambda v1, v2: v1 + v2,
                      map(lambda r: np.maximum(rCon[z][r] * nt_z - rCap[r], 0), [r for r in range(totalResource)]))
    tmp = np.zeros((totalZONES))
    for z in planningZONES:
        if z not in dummyNbrs:
            tmp[z] = count(z, nt_z[z])
        else:
            tmp[z] = 0
    return tmp
    # return map(lambda z : float(tmp[z])/rCap[z], [r for r in range(len(planningZONES))])

def getDelay(nt_z):
    def cost(z, nt_z):
        # ------ With Delay Cost --- #
        cost = 0
        if nt_z == 0:
            cost = 0
        else:
            cost = wDelay
        return float(nt_z * cost)

    tmp = np.zeros((totalZONES))
    for z in planningZONES:
        tmp[z] = 1 * cost(z, nt_z[z])
        # if z not in dummyNbrs:
        #     tmp[z] = 1 * cost(z, nt_z[z])
        # else:
        #     tmp[z] = 0
    return tmp

def getResVio(nt_z, zGraph):

    def costRes(z, nt_z):
        # ------ With Resource Violation ---- #
        cost = 0
        cost += reduce(lambda v1, v2: v1 + v2, map(lambda r: np.maximum((rCon[z][r] * nt_z - rCap[r]), 0),
                                                   [r for r in range(totalResource)]))
        return float(wResource * cost)

    tmp = np.zeros((totalZONES))
    for z in planningZONES:
        if z not in dummyNbrs:
            tmp[z] = 1 * costRes(z, nt_z[z])
        else:
            tmp[z] = 0
    return tmp

def eta(t_p, betaH, n_s_sp_t, zGraph, Zones):

    tSum = 0
    for z in planningZONES:
        for tau in range(1, t_p+1):
            zp_list = filter(lambda ztmp : ztmp <> z, [ _ for _ in range(totalZONES)])
            v1 = map(lambda zp : n_s_sp_t[z][tau][zp][t_p+1], zp_list)
            v2 = map(lambda zp : np.log(betaH[tau][z][zp]), zp_list)
            temp1 = np.matmul(v1, v2)
            temp2 = n_s_sp_t[z][tau][z][tau] * np.log(1 - p_nbr(z, betaH[tau], zGraph, Zones))
            tSum += (temp1 + temp2)
    return tSum

def p_nbr(z, beta, zGraph, Zones):

    tmpPzz = map(lambda zp: Pzz(z, zp, zGraph), Zones[z].nbr)
    # ------- Normalize ---- #
    # better to normalize before on Pzz()
    sum = np.sum(tmpPzz)
    tmpPzz = map(lambda zp: zp / sum, tmpPzz)
    tmpBeta = map(lambda zp: beta[z][zp], Zones[z].nbr)
    return np.matmul(tmpPzz, tmpBeta)

def p_nbrTF(z, beta, zGraph, Zones):

    tmpPzz = map(lambda zp: Pzz(z, zp, zGraph), Zones[z].nbr)
    # ------- Normalize ---- #
    # better to normalize before on Pzz()
    sum = tf.reduce_sum(tmpPzz)
    tmpPzz = map(lambda zp: zp / sum, tmpPzz)
    tmpBeta = map(lambda zp: beta[z][zp], Zones[z].nbr)
    return tf.reduce_sum(tf.multiply(tmpPzz, tmpBeta))

def displayCount(t, nt):

    zone_count = updateZoneCount(t, nt)
    print t, zone_count

def updateZoneCount(tau, cT, zGraph):

    zone_count = np.zeros(totalZONES)
    for z in range(totalZONES):
        count = 0
        for zp in nx.neighbors(zGraph, z):
            for t in range(tau+1, HORIZON+1):
                count += cT.nt_zz[z][zp][t]
        zone_count[z] = count + cT.nt_zt[z][tau]
    return zone_count

def logic(t, z_next, tau_next, z, tau):

    def modusPonen(p, q):
        return (not p) or q

    # ------- Cond1 ------- #
    p = (z == z_next)
    q = (tau == tau_next)
    cond1 = modusPonen(p, q)

    # ------- Cond2 ------- #
    p = (z <> z_next)
    q = (tau_next == t+1)
    cond2 = modusPonen(p, q)

    # ------ Cond3 ------- #
    p = (z == dummyZONES)
    q = (tau == 0)
    cond3 = modusPonen(p, q)
    return cond1 and cond2 and cond3

def runningAvg(index, prev, sample):
    if index == 1:
        return sample
    else:
        return (prev * index + sample) / (index + 1)

def getMaxReward():


    cost = 0
    cost += reduce(lambda v1, v2: v1 + v2, map(lambda r:np.maximum((rCon[1][r] * totalVESSELS - rCap[r]), 0), [r for r in range(totalResource)]))

    return wResource * cost + wDelay

def cost_t_nt_z(z, nt_z):

    costRes = 0
    if z not in dummyNbrs:
        # ------ With Resource Violation ---- #
        costRes += reduce(lambda v1, v2: v1 + v2, map(lambda r: np.maximum((rCon[z][r] * nt_z - rCap[r]), 0), [r for r in range(totalResource)]))

    costDelay = 0
    if nt_z == 0:
        costDelay = 0
    else:
        costDelay = wDelay

    return float( wResource * costRes +  costDelay)

def updateCountTable(cT_new, cT):

    cT.nt_zt = cT_new.nt_zt
    cT.nt_zz = cT_new.nt_zz
    cT.nt_zt_zt = cT_new.nt_zt_zt
    cT.nt_z = cT_new.nt_z
    cT.ntz_zt = cT_new.ntz_zt
    return cT

def dump():

    maps = [0, 1, 2, 3]
    dirs = ['logs', 'tboard']

    for m in maps:
        for d in dirs:
            remote = "projects/collective"+str(m)+"_1/"+d
            # local = "/media/james/Storage/Dropbox/Buffer/project_results/collective/Experiment_v0/Delay_Resource/RL/rCap_0.5/m"+str(m)
            local = "/Users/james/Dropbox/Buffer/project_results/collective/Experiment_v0/Delay_Resource/RL/rCap_0.8/m"+str(m)

            cmd = "dw.py -H -d "+remote+" "+local
            os.system(cmd)

def main():

    dump()

# =============================================================================== #


if __name__ == '__main__':
    main()
