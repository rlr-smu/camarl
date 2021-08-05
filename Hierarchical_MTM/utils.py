
# ================================ Imports ================================ #

import numpy as np
# from data import totalZONES, dummyZONES, HORIZON, totalResource, rCap, rCon, planningZONES, termZONES, Pzz, totalVESSELS, T_min_max, zGraph, dummyNbrs
from parameters import wDelay, wResource, MAP_ID, DISCOUNT, HORIZON, TOTAL_VESSEL, NUM_OPTIONS, BUFFER_VIO
from data import env_data
import pdb
import rlcompleter
import networkx as nx
from auxLib import average
from collections import Counter
from torch.distributions import Categorical
import torch as tc
from scipy.stats import binom
import math
from functools import reduce
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #
dt = env_data(mapName=MAP_ID)
planningZONES = dt.planningZONES
zGraph = dt.zGraph
totalResource = dt.totalResource
rCon = dt.rCon
totalVESSELS = TOTAL_VESSEL
rCap = dt.rCap
dummyNbrs = dt.dummyNbrs
totalZONES = dt.totalZONES
T_min_max = dt.T_min_max
# ============================================================================ #

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def computeDistStats(n, p):

    def mode(n, p):
        if p == 0:
            return 0
        elif p == 1:
            return n
        else:
            return math.floor((n+1)*p)
    dist = binom(n, p)
    mean = dist.mean()
    median = dist.median()
    return round(mean, 2), int(median), int(mode(n,p))

def display_xi_entr_t_eval(o_selected=None, xi_all=None, nt_z=None, xi_avg=None, xi_avg_all=None, pi_avg=None, dist_stats=None):
    xiStr = ""
    for z in planningZONES:
        for zp in nx.neighbors(zGraph, z):
            o = o_selected[z][zp]
            if nt_z[z] > 0:
                xi_avg[(z, zp)].append(round(xi_all[z][zp][o], 2))
                pi_avg[(z, zp)].append(o)
                # ent_avg[(z, zp)].append(float(ent))
                for i in range(NUM_OPTIONS):
                    xi_avg_all[(z, zp, i)].append(round(xi_all[z][zp][i], 2))
                    p = xi_all[z][zp][i]
                    tmin = T_min_max[z][zp][0]
                    tmax = T_min_max[z][zp][1]
                    n = tmax - tmin
                    mean, median, mode = computeDistStats(n, p)
                    mean_real = round(tmin + mean, 2)
                    med_real = tmin + median
                    mode_real = tmin+mode

                    dist_stats[(z, zp, i)]['mean'].append(mean_real)
                    dist_stats[(z, zp, i)]['median'].append(med_real)
                    dist_stats[(z, zp, i)]['mode'].append(mode_real)
                dist = str(dist_stats[(z, zp, o)]['mean'][-1])+","+str(dist_stats[(z, zp, o)]['median'][-1])+","+str(dist_stats[(z, zp, o)]['mode'][-1])
            else:
                dist = "-1,-1,-1"
            xiStr += str(o)+ "_xi_" + str(z) +"_"+ str(zp) + " : " + str(round(xi_all[z][zp][o], 2)) + " stats : "+dist+"  "
    return xiStr, xi_avg, pi_avg, xi_avg_all, dist_stats

def display_xi_entr_t_dist(o_selected=None, xi_all=None, nt_z=None, xi_avg=None, xi_avg_all=None, pi_avg=None, pi=None, ent_avg=None, dist_stats=None):
    xiStr = ""
    for z in planningZONES:
        for zp in nx.neighbors(zGraph, z):
            o = o_selected[z][zp]
            pi_zzp = pi[z][zp]
            ent = Categorical(tc.tensor(pi_zzp)).entropy().numpy()
            if nt_z[z] > 0:
                xi_avg[(z, zp)].append(round(xi_all[z][zp][o], 2))
                pi_avg[(z, zp)].append(o)
                ent_avg[(z, zp)].append(float(ent))
                for i in range(NUM_OPTIONS):
                    xi_avg_all[(z, zp, i)].append(round(xi_all[z][zp][i], 2))
                    p = xi_all[z][zp][i]
                    tmin = T_min_max[z][zp][0]
                    tmax = T_min_max[z][zp][1]
                    n = tmax - tmin
                    mean, median, mode = computeDistStats(n, p)
                    mean_real = round(tmin + mean, 2)
                    med_real = tmin + median
                    mode_real = tmin+mode

                    dist_stats[(z, zp, i)]['mean'].append(mean_real)
                    dist_stats[(z, zp, i)]['median'].append(med_real)
                    dist_stats[(z, zp, i)]['mode'].append(mode_real)
                dist = str(dist_stats[(z, zp, o)]['mean'][-1])+","+str(dist_stats[(z, zp, o)]['median'][-1])+","+str(dist_stats[(z, zp, o)]['mode'][-1])
            else:
                dist = "-1,-1,-1"


            # print(xi_all[z][zp][o])
            # print(round(xi_all[z][zp][o], 2))
            # print(ent)            
            # print(str(ent)[0:5])
            # print(round(ent, 2))
            # print(dist)

            # xiStr += str(o)+ "_xi_" + str(z) +"_"+ str(zp) + " : " + str(round(xi_all[z][zp][o], 2)) + " ent : "+str(round(ent, 2))+"  stats : "+dist+"  "

            xiStr += str(o)+ "_xi_" + str(z) +"_"+ str(zp) + " : " + str(round(xi_all[z][zp][o], 2)) + " ent : "+str(str(ent)[0:5])+"  stats : "+dist+"  "


    return xiStr, xi_avg, pi_avg, xi_avg_all, ent_avg, dist_stats

def display_xi_entr_t(o_selected=None, xi_all=None, nt_z=None, xi_avg=None, xi_avg_all=None, pi_avg=None, pi=None, ent_avg=None):
    xiStr = ""
    for z in planningZONES:
        for zp in nx.neighbors(zGraph, z):
            o = o_selected[z][zp]
            pi_zzp = pi[z][zp]
            ent = Categorical(tc.tensor(pi_zzp)).entropy().numpy()
            if nt_z[z] > 0:
                xi_avg[(z, zp)].append(round(xi_all[z][zp][o], 2))
                pi_avg[(z, zp)].append(o)
                ent_avg[(z, zp)].append(float(ent))
                for i in range(NUM_OPTIONS):
                    xi_avg_all[(z, zp, i)].append(round(xi_all[z][zp][i], 2))
            xiStr += str(o)+ "_xi_" + str(z) +"_"+ str(zp) + " : " + str(round(xi_all[z][zp][o], 2)) + " ent : "+str(round(ent, 2))+"  "
    return xiStr, xi_avg, pi_avg, xi_avg_all, ent_avg

def display_xi_t(o_selected=None, xi_all=None, nt_z=None, xi_avg=None, xi_avg_all=None, pi_avg=None):
    xiStr = ""
    for z in planningZONES:
        for zp in nx.neighbors(zGraph, z):
            o = o_selected[z][zp]
            if nt_z[z] > 0:
                # pdb.set_trace()
                xi_avg[(z, zp)].append(round(xi_all[z][zp][o], 2))
                pi_avg[(z, zp)].append(o)

                for i in range(NUM_OPTIONS):
                    xi_avg_all[(z, zp, i)].append(round(xi_all[z][zp][i], 2))

            xiStr += str(o)+ "_xi_" + str(z) +"_"+ str(zp) + " : " + str(round(xi_all[z][zp][o], 2)) + " "
    return xiStr, xi_avg, pi_avg, xi_avg_all

def display_avg_eval(xi_avg=None, pi_avg=None, ax=None, xi_avg_all=None, dist_stats=None):

    xiStr = ""
    opStr = ""
    xi_avg2 = {}
    dist = {}
    xi_all_avg3 = {}
    dist_stats_avg = {}
    dist_stats_avg2 = {}
    statStr = ""
    flag = True
    for z in planningZONES:
        zp_nbr = nx.neighbors(zGraph, z)
        for zp in zp_nbr:
            if flag:
                statStr += str(z) + "_" + str(zp) + " | "
                flag = False
            else:
                statStr += "    "+str(z) + "_" + str(zp) + " | "
            opStr += str(z) + "_" + str(zp)+" | "
            dist[(z, zp)] = {}
            dist_stats_avg[(z, zp)] = {}
            dist_stats_avg2[(z, zp)] = {}
            xi_all_avg3[(z, zp)] = {}
            # ------op
            t1 = pi_avg[(z, zp)]
            t2 = dict(Counter(t1))
            for o in range(NUM_OPTIONS):
                xi_all_avg3[(z, zp)][o] = round(average(xi_avg_all[(z, zp, o)]), 2)
                if (z, zp, o) in dist_stats:
                    dist_stats_avg2[(z, zp)][o] = []
                    l1 = np.array(dist_stats[(z, zp, o)]['mean'])
                    mn = l1.mean()
                    std = l1.std()
                    dist_stats_avg2[(z, zp)][o].append((round(mn, 2), round(std, 2)))
                    # dist_stats_avg2[(z, zp)][o].append(round(average(dist_stats[(z, zp, o)]['median']), 2))
                    l2 = np.array(dist_stats[(z, zp, o)]['mode'])
                    mn = l2.mean()
                    std = l2.std()
                    dist_stats_avg2[(z, zp)][o].append((round(mn, 2), round(std, 2)))
                    # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
                    # pdb.set_trace()

                if o in t2:
                    dist[(z, zp)][o] = round(float(t2[o])/len(t1), 2)
                else:
                    dist[(z, zp)][o] = 0.0
            statStr += str(dist_stats_avg2[(z, zp)]) + "\n"
            ax.writeln("      " + str(z) + "_" + str(zp)+" : "+str(dist[(z,zp)])+ " | "+str(xi_all_avg3[(z,zp)]))
            xi_avg2[(z,zp)] = round(average(xi_avg[(z, zp)]), 2)
            xiStr += "xi" + str(z) +"_"+ str(zp) + " : " + str(xi_avg2[(z,zp)]) + " "
    return  xiStr, xi_avg2, dist, xi_all_avg3, statStr, dist_stats_avg2

def display_avg_ent_dist(xi_avg=None, pi_avg=None, ax=None, xi_avg_all=None, ent_avg=None, dist_stats=None):

    xiStr = ""
    opStr = ""
    entStr = ""
    xi_avg2 = {}
    ent_avg2 = {}
    dist = {}
    xi_all_avg3 = {}
    dist_stats_avg = {}
    dist_stats_avg2 = {}
    statStr = ""
    flag = True
    for z in planningZONES:
        zp_nbr = nx.neighbors(zGraph, z)
        for zp in zp_nbr:
            if flag:
                statStr += str(z) + "_" + str(zp) + " | "
                flag = False
            else:
                statStr += "    "+str(z) + "_" + str(zp) + " | "
            opStr += str(z) + "_" + str(zp)+" | "
            dist[(z, zp)] = {}
            dist_stats_avg[(z, zp)] = {}
            dist_stats_avg2[(z, zp)] = {}
            xi_all_avg3[(z, zp)] = {}
            # ------op
            t1 = pi_avg[(z, zp)]
            t2 = dict(Counter(t1))
            for o in range(NUM_OPTIONS):
                xi_all_avg3[(z, zp)][o] = round(average(xi_avg_all[(z, zp, o)]), 2)
                if (z, zp, o) in dist_stats:
                    dist_stats_avg2[(z, zp)][o] = []
                    l1 = np.array(dist_stats[(z, zp, o)]['mean'])
                    mn = l1.mean()
                    std = l1.std()
                    dist_stats_avg2[(z, zp)][o].append((round(mn, 2), round(std, 2)))
                    l2 = np.array(dist_stats[(z, zp, o)]['mode'])
                    mn = l2.mean()
                    std = l2.std()
                    dist_stats_avg2[(z, zp)][o].append((round(mn, 2), round(std, 2)))
                if o in t2:
                    dist[(z, zp)][o] = round(float(t2[o])/len(t1), 2)
                else:
                    dist[(z, zp)][o] = 0.0
            statStr += str(dist_stats_avg2[(z, zp)]) + "\n"
            ax.writeln("      " + str(z) + "_" + str(zp)+" : "+str(dist[(z,zp)])+ " | "+str(xi_all_avg3[(z,zp)]))
            xi_avg2[(z,zp)] = round(average(xi_avg[(z, zp)]), 2)
            ent_avg2[(z, zp)] = round(average(ent_avg[(z, zp)]), 2)
            xiStr += "xi" + str(z) +"_"+ str(zp) + " : " + str(xi_avg2[(z,zp)]) + " "
            entStr += "ent "+str(z)+"_"+ str(zp) + " : " + str(ent_avg2[(z, zp)]) + " "
    return  xiStr, xi_avg2, dist, xi_all_avg3, ent_avg2, entStr, statStr, dist_stats_avg2

def display_avg_ent(xi_avg=None, pi_avg=None, ax=None, xi_avg_all=None, ent_avg=None):

    xiStr = ""
    opStr = ""
    entStr = ""
    xi_avg2 = {}
    ent_avg2 = {}
    dist = {}
    xi_all_avg3 = {}
    for z in planningZONES:
        zp_nbr = nx.neighbors(zGraph, z)
        for zp in zp_nbr:
            opStr += str(z) + "_" + str(zp)+" | "
            dist[(z, zp)] = {}
            xi_all_avg3[(z, zp)] = {}
            # ------op
            t1 = pi_avg[(z, zp)]
            t2 = dict(Counter(t1))
            for o in range(NUM_OPTIONS):
                xi_all_avg3[(z, zp)][o] = round(average(xi_avg_all[(z, zp, o)]), 2)
                if o in t2:
                    dist[(z, zp)][o] = round(float(t2[o])/len(t1), 2)
                else:
                    dist[(z, zp)][o] = 0.0
            ax.writeln("      " + str(z) + "_" + str(zp)+" : "+str(dist[(z,zp)])+ " | "+str(xi_all_avg3[(z,zp)]))
            xi_avg2[(z,zp)] = round(average(xi_avg[(z, zp)]), 2)
            ent_avg2[(z, zp)] = round(average(ent_avg[(z, zp)]), 2)
            xiStr += "xi" + str(z) +"_"+ str(zp) + " : " + str(xi_avg2[(z,zp)]) + " "
            entStr += "ent "+str(z)+"_"+ str(zp) + " : " + str(ent_avg2[(z, zp)]) + " "
    return  xiStr, xi_avg2, dist, xi_all_avg3, ent_avg2, entStr

def getDiscountedReturn(tmpRew):

    tmpRew2 = []
    for t in range(len(tmpRew)):
        tmpRew2.append(tmpRew[t]*pow(DISCOUNT, t))
    return sum(tmpRew2)

def getMaxReward():
    cost = 0
    cost += reduce(lambda v1, v2: v1 + v2, map(lambda r:np.maximum((rCon[1][r] * totalVESSELS - rCap[r]), 0), [r for r in range(totalResource)]))
    return wResource * cost + wDelay

def cost_t_nt_z(z, nt_z):

    costRes = 0

    if BUFFER_VIO:
        costRes += reduce(lambda v1, v2: v1 + v2, map(lambda r: np.maximum((rCon[z][r] * nt_z - rCap[r]), 0), [r for r in range(totalResource)]))
    else:
        if z not in dummyNbrs:
            # ------ With Resource Violation ---- #
            costRes += reduce(lambda v1, v2: v1 + v2, map(lambda r: np.maximum((rCon[z][r] * nt_z - rCap[r]), 0), [r for r in range(totalResource)]))

    costDelay = 0
    if nt_z == 0:
        costDelay = 0
    else:
        costDelay = wDelay
    resVio = wResource * costRes
    delay = costDelay
    return float(resVio+delay), resVio, delay

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

def getTravelDelay(t, cT, zGraph):

    tTravel = {}
    for z in planningZONES:
        timeSum = 0
        for zp in nx.neighbors(zGraph, z):
            tMin = T_min_max[z][zp][0]
            tMax = T_min_max[z][zp][1]
            tp_space = [_ for _ in range(t + tMin, t + tMax + 1)]
            tp_space = filter(lambda tp2: tp2 <= HORIZON, tp_space)

            # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
            # pdb.set_trace()

            tmp2 = list(map(lambda tp: (tp - t - tMin) * cT.nt_zt_zt[z][t][zp][tp], tp_space))
            timeSum += np.sum(tmp2)
        if np.sum(cT.nt_zt_zt[z][t]) > 0:
            finalSum = timeSum / np.sum(cT.nt_zt_zt[z][t])
        else:
            finalSum = timeSum
        tTravel[z] = finalSum
    return average(tTravel.values())
