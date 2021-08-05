"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 13 Apr 2019
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
from parameters import ENV_VAR, OPENBLAS, OMP, MKL, NUMEXPR, SAMPLE_AVG, TOTAL_RUNTIME, EPISODES, SHOULD_LOG, BATCH_SIZE, LOAD_MODEL, wDelay, wResource, capRatio, succProb, MODES, ARR_HORIZON, NUM_OPTIONS, MAP_ID, HORIZON, TOTAL_VESSEL, EVAL_EPISODES, SAVE_MODEL, BUFFER_CAP, BUFFER_VIO

if ENV_VAR:
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS)
    os.environ["OMP_NUM_THREADS"] = str(OMP)
    os.environ["MKL_NUM_THREADS"] = str(MKL)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUMEXPR)

from data import env_data
import networkx as nx
import numpy as np

# from environment_test import Maritime, countTable
# from utils_test import getVioCount, getTravelDelay, getDiscountedReturn

from environment import Maritime, countTable
from utils import getVioCount, getTravelDelay, getDiscountedReturn, display_xi_t, display_xi_entr_t,display_xi_entr_t_eval, display_avg_eval, display_avg_ent, computeDistStats, display_xi_entr_t_dist, display_avg_ent_dist

from agents.op_ind.agent import op_ind
from agents.op_pgv.agent import op_pgv

from scipy.stats import binom
from copy import deepcopy
import pdb
import rlcompleter
from tqdm import tqdm
# from dashBrd import drawOption
import auxLib as ax
from auxLib import average, file2list

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, help='Agent to run')
args = parser.parse_args()

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #

dirName = str(MAP_ID) + "_" + str(wDelay) + "_" + str(wResource) + "_" + str(capRatio) + "_" + str(
    HORIZON) + "_" + str(TOTAL_VESSEL) + "_" + args.agent

ax.opFile = ""
if LOAD_MODEL:
    ax.opFile = "./log/" + dirName + "/" + dirName + "_Inference" + ".txt"
else:
    ax.opFile = "./log/" + dirName + "/" + dirName + ".txt"


# ============================================================================ #

def init():

    os.system("mkdir ./log/")
    os.system("mkdir ./log/" + dirName)

    os.system("rm " + ax.opFile)
    ax.deleteDir("./log/" + dirName + "/plots/")

    os.system("cp parameters.py ./log/"+dirName+"/")
    os.system("mkdir ./log/"+dirName+"/plots")

    ax.deleteDir("./runs/")
    os.system("mkdir ./log/" + dirName + "/model")

def initDisplay(agent, data):

    ax.writeln("------------------------------")
    ax.writeln("Agent : " + agent)
    ax.writeln("Map : "+str(MAP_ID))
    ax.writeln("HORIZON : " + str(HORIZON))
    ax.writeln("succProb : " + str(succProb))
    ax.writeln("Total Vessel : " + str(TOTAL_VESSEL))
    ax.writeln("Total Zones : " + str(data.totalZONES))
    ax.writeln("Planning Zones : " + str(data.planningZONES))
    ax.writeln("Start Zones : " + str(data.dummyZONES))
    ax.writeln("Terminal Zones : " + str(data.termZONES))
    ax.writeln("Total Resources : " + str(data.totalResource))
    ax.writeln("Delay Weight : " + str(wDelay))
    ax.writeln("Resource Weight : " + str(wResource))
    ax.writeln("Resource Capacity Ratio : " + str(float(capRatio)))
    ax.writeln("Resource Capacity : " + str(data.rCap))
    ax.writeln("Arrival Modes : " + str(MODES))
    ax.writeln("Arrival Horizon : " + str(ARR_HORIZON))
    ax.writeln("Buffer Cap : " + str(BUFFER_CAP))
    ax.writeln("Buffer Vio : " + str(BUFFER_VIO))
    ax.writeln("------------------------------")

def evaluate(data, env, agent):


    ax.opFile = "./log/" + dirName + "/" + dirName + "_eval" + ".txt"
    os.system("rm " + ax.opFile)
    tStart = ax.now()
    runTime = 0
    buffer_Return = np.zeros(EVAL_EPISODES)
    buffer_Vio = np.zeros(EVAL_EPISODES)
    buffer_Delay = np.zeros(EVAL_EPISODES)
    buffer_VioPenalty = np.zeros(EVAL_EPISODES)
    buffer_DelayPenalty = np.zeros(EVAL_EPISODES)
    totalZONES = data.totalZONES
    opt_freq = {}
    tm1 = []
    for z in data.planningZONES:
        for zp in nx.neighbors(data.zGraph, z):
            opt_freq[(z, zp)] = {}
            for o in range(NUM_OPTIONS):
                opt_freq[(z, zp)][o] = {}

    ax.writeln("----------------------------------------- EVALUATION STARTS ------------------------------------------")
    for i_episode in range(1, EVAL_EPISODES+1):

        # ------- Init Environment
        cT = env.init()

        tStart2 = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("-----------------------------------------------------------------------------------")

            ax.writeln("> Eval Episode : "+ str(i_episode))
            xi_avg = {}
            xi_avg_all = {}
            distStats_avg = {}
            ent_avg = {}
            pi_avg = {}
            for z in data.planningZONES:
                for zp in nx.neighbors(data.zGraph, z):
                    for o in range(NUM_OPTIONS):
                        xi_avg_all[(z, zp, o)] = []
                        distStats_avg[(z, zp, o)] = {}
                        distStats_avg[(z, zp, o)]['mean'] = []
                        distStats_avg[(z, zp, o)]['median'] = []
                        distStats_avg[(z, zp, o)]['mode'] = []
                    xi_avg[(z, zp)] = []
                    pi_avg[(z, zp)] = []
                    ent_avg[(z, zp)] = []
        epRewardList = np.zeros(HORIZON+1)
        epVioCountList = np.zeros(HORIZON+1)
        epTrDelayList = np.zeros(HORIZON+1)
        epVioPen = np.zeros(HORIZON+1)
        epDelayPen = np.zeros(HORIZON+1)
        epCount = 0
        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        for t in range(HORIZON):
            o, pi, xi_all = agent.getOption(cT)
            buffer_nt_z[t] = cT.nt_z
            Rt, _, cT_new,  vioPen, delayPen  = env.step(t, cT, xi_all, pi)
            for z in data.planningZONES:
                for zp in nx.neighbors(data.zGraph, z):
                    t1 = cT_new.nt_z_o[z][zp]
                    if np.sum(cT_new.nt_z_o[z][zp]) > 0 and len(t1[t1 > 0]) > 1:
                        tm1.append(i_episode)
                        for k in range(NUM_OPTIONS):
                            if cT_new.nt_z_o[z][zp][k] > 0:
                                if i_episode not in opt_freq[(z, zp)][k]:
                                    opt_freq[(z, zp)][k][i_episode] = []
                                opt_freq[(z, zp)][k][i_episode].append(cT_new.nt_z_o[z][zp][k])
            # ------- Other Stats ------ #
            vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
            trDelay = getTravelDelay(t, cT_new, data.zGraph)
            epRewardList[epCount] = Rt
            epVioCountList[epCount] = vioCount
            epTrDelayList[epCount] = trDelay
            epVioPen[epCount] = vioPen
            epDelayPen[epCount] = delayPen
            epCount += 1
            if i_episode % SHOULD_LOG == 0:
                # -------- Xi
                xiStr = ""
                if t > 0 :
                    xiStr, xi_avg, pi_avg, xi_avg_all, distStats_avg = display_xi_entr_t_eval(o_selected=o, xi_all=xi_all, nt_z=cT.nt_z, xi_avg=xi_avg, pi_avg=pi_avg, xi_avg_all=xi_avg_all, dist_stats=distStats_avg)
                ax.writeln("  " + str(t) + " " + str(Rt) + " " + str(buffer_nt_z[t])+" | "+xiStr)
            cT = cT_new
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))
        epReward = getDiscountedReturn(epRewardList)
        epVioPenDiscount = round(getDiscountedReturn(epVioPen), 3)
        epDelayPenDiscount = round(getDiscountedReturn(epDelayPen), 3)
        buffer_Return[i_episode-1] = epReward
        buffer_VioPenalty[i_episode-1] = epVioPenDiscount
        buffer_DelayPenalty[i_episode-1] = epDelayPenDiscount
        buffer_Vio[i_episode-1] = np.sum(epVioCountList)
        buffer_Delay[i_episode-1] = np.sum(epTrDelayList)
        if runTime > TOTAL_RUNTIME:
            break
        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            epVioCount = np.sum(epVioCountList) # average(epVioCountList)
            epTrDelay = np.sum(epTrDelayList)
            ax.writeln("\n\n  Total Episode Reward : "+ str(round(epReward, 2)))
            ax.writeln("  Total Episode VioPenalty : " + str(round(epVioPenDiscount, 2)))
            ax.writeln("  Total Episode DelayPenalty : " + str(round(epDelayPenDiscount, 2)))
            ax.writeln("  ")
            ax.writeln("  Total Res. Vio : " + str(round(np.sum(epVioCountList), 2)))
            ax.writeln("  Total Travel Delay : " + str(round(np.sum(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Avg. Res. Vio : " + str(round(np.average(epVioCountList), 2))+" std : "+str(round(np.std(epVioCountList), 2)))
            ax.writeln("  Avg. Travel Delay : " + str(round(np.average(epTrDelayList), 2))+" std : "+str(round(np.std(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Max Res. Vio : " + str(round(np.max(epVioCountList), 2)))
            ax.writeln("  Max Travel Delay : " + str(round(np.max(epTrDelayList), 2)))

            ax.writeln("\n  Option Stats:")
            ax.writeln("      Distribution    |    Values | Stats")
            xiStr, xi_avg2, opDist, xi_all_avg2, statsStr, dist_stats2 = display_avg_eval(pi_avg=pi_avg, xi_avg=xi_avg, ax=ax, xi_avg_all=xi_avg_all, dist_stats=distStats_avg)
            ax.writeln("\n  Average Xi : ")
            ax.writeln("    " + xiStr)
            ax.writeln("\n  Travel Time Distribution Stats : ")
            ax.writeln("    " + statsStr)
            tstr = "  Mode # "
            for z in data.planningZONES:
                for zp in nx.neighbors(data.zGraph, z):
                    tstr += str(z) + "_" + str(zp)
                    tstr2 = ""
                    for k in range(NUM_OPTIONS):
                        tstr2 += " : "+str(dist_stats2[(z, zp)][k][1][0])+","+str(dist_stats2[(z, zp)][k][1][1])
                    tstr += tstr2+" | "
            ax.writeln(tstr[:-3])

            opt_freq_ep = {}
            for z in data.planningZONES:
                for zp in nx.neighbors(data.zGraph, z):
                    opt_freq_ep[(z, zp)] = {}
                    for o in range(NUM_OPTIONS):
                        if i_episode in opt_freq[(z, zp)][o]:
                            t1 = np.array(opt_freq[(z, zp)][o][i_episode])
                            opt_freq_ep[(z, zp)][o] = (round(np.mean(t1),2), round(np.std(t1), 2))
            tstr = "OptFreq # "
            for (z, zp) in opt_freq_ep:
                tstr += str(z)+"_"+str(zp) +" : "
                for k in opt_freq_ep[(z, zp)]:
                    tstr += str(opt_freq_ep[(z, zp)][k][0])+","+str(opt_freq_ep[(z, zp)][k][1]) +" : "
                tstr = tstr[:-3]
                tstr += " | "
            tstr = tstr[:-3]
            ax.writeln("\n\n  "+tstr)
            ax.writeln("\n  Runtime : " + str(round(runTime, 3)) + " Seconds")

        tEnd = ax.now()
        runTime += ax.getRuntime(tStart2, tEnd)

    # ----------- Option Freq -------- #
    opt_freq_sum = {}
    for z in data.planningZONES:
        for zp in nx.neighbors(data.zGraph, z):
            opt_freq_sum[(z, zp)] = {}
            for o in range(NUM_OPTIONS):
                opt_freq_sum[(z, zp)][o] = {}
                for i in range(1, EVAL_EPISODES+1):
                    if i in opt_freq[(z, zp)][o]:
                        t1 = np.array(opt_freq[(z, zp)][o][i])
                        opt_freq[(z, zp)][o].pop(i)
                        opt_freq[(z, zp)][o][i] = np.mean(t1)
                        opt_freq_sum[(z, zp)][o][i] = np.sum(t1)

    for z in data.planningZONES:
        for zp in nx.neighbors(data.zGraph, z):
            for o in range(NUM_OPTIONS):
                t1 = np.array(opt_freq[(z, zp)][o].values())
                opt_freq[(z, zp)].pop(o)
                opt_freq[(z, zp)][o] = (round(np.mean(t1), 2), round(np.std(t1), 2))
                opt_freq_sum[(z, zp)].pop(o)
                opt_freq_sum[(z, zp)][o] = round(np.sum(t1), 2)

    ax.writeln("----------------------------------------------------------------------------------------")
    max_return = max(buffer_Return)
    indx = np.where(buffer_Return==max_return)[0][0]
    max_vioPen = round(buffer_VioPenalty[indx], 2)
    max_delayPen = round(buffer_DelayPenalty[indx], 2)
    max_Vio = round(buffer_Vio[indx], 2)
    max_Delay = round(buffer_Delay[indx], 2)
    ax.writeln("\n\n  Best Episode No. : " + str(indx+1))
    ax.writeln("  Best Episode Return : "+ str(max_return))
    ax.writeln("  Best Episode VioPenalty : " + str(max_vioPen))
    ax.writeln("  Best Episode DelayPenalty : " + str(max_delayPen))
    ax.writeln("  Best Res. Vio : " + str(max_Vio))
    ax.writeln("  Best Travel Delay : " + str(max_Delay))
    tstr = "OptFreq_Total # "
    for (z, zp) in opt_freq:
        tstr += str(z) + "_" + str(zp) + " : "
        for k in opt_freq[(z, zp)]:
            tstr += str(opt_freq[(z, zp)][k][0]) + "," + str(opt_freq[(z, zp)][k][1]) + " : "
        tstr = tstr[:-3]
        tstr += " | "
    tstr = tstr[:-3]
    ax.writeln("\n\n  " + tstr)
    ax.writeln("----------------------------------------- EVALUATION ENDS ------------------------------------------")
    totalRunTime = ax.getRuntime(tStart, ax.now())
    ax.writeln("Total Runtime : " + str(round(totalRunTime, 3)) + " Seconds")
    ax.writeln("Evaluation Complete !")
    ax.writeln(str(tm1))

def main():

    tStart = ax.now()
    runTime = 0
    init()
    metrics = ['reward', 'delay', 'vio', 'q_loss', 'beta_loss', 'total_loss']

    data = env_data(mapName=MAP_ID)
    env = Maritime(data=data)

    if args.agent == "op_ind":
        agent = op_ind(data=data, metrics=metrics, dirName=dirName)
    elif args.agent == "op_pgv":
        agent = op_pgv(data=data, metrics=metrics, dirName=dirName)
    else:
        print ("Error: Agent not defined!")

    tEnd = ax.now()
    runTime += ax.getRuntime(tStart, tEnd)
    #-----------------
    buffer_runAvg = np.zeros(SAMPLE_AVG)
    buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
    buffer_trDelayAvg = np.zeros(SAMPLE_AVG)

    buffer_VioPenalty = np.zeros(SAMPLE_AVG)
    buffer_DelayPenalty = np.zeros(SAMPLE_AVG)

    sampleID = 0
    batchID = 0
    totalZONES = data.totalZONES
    initDisplay(args.agent, data)
    tm1 = []
    opt_freq = {}
    for z in data.planningZONES:
        for zp in nx.neighbors(data.zGraph, z):
            opt_freq[(z, zp)] = {}
            for o in range(NUM_OPTIONS):
                opt_freq[(z, zp)][o] = {}

    for i_episode in range(1, EPISODES+1):

        # ------- Init environment
        cT = env.init()

        batchID += 1
        tStart2 = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("-----------------------------------------------------------------------------------")
            ax.writeln("> Episode : "+ str(i_episode))
            xi_avg = {}
            xi_avg_all = {}
            pi_avg = {}
            ent_avg = {}
            distStats_avg = {}

            for z in data.planningZONES:
                for zp in nx.neighbors(data.zGraph, z):
                    for o in range(NUM_OPTIONS):
                        xi_avg_all[(z, zp, o)] = []
                        distStats_avg[(z, zp, o)] = {}
                        distStats_avg[(z, zp, o)]['mean'] = []
                        distStats_avg[(z, zp, o)]['median'] = []
                        distStats_avg[(z, zp, o)]['mode'] = []

                    xi_avg[(z, zp)] = []
                    pi_avg[(z, zp)] = []
                    ent_avg[(z, zp)] = []
        epRewardList = np.zeros(HORIZON+1)
        epVioCountList = np.zeros(HORIZON+1)
        epTrDelayList = np.zeros(HORIZON+1)

        epVioPen = np.zeros(HORIZON+1)
        epDelayPen = np.zeros(HORIZON+1)

        epCount = 0
        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        buffer_rt = np.zeros(HORIZON)
        buffer_rt_z = np.zeros((HORIZON, totalZONES))
        buffer_nt_z_o = np.zeros((HORIZON, totalZONES, totalZONES, NUM_OPTIONS))
        buffer_nt_z_o_tau = np.zeros((HORIZON, totalZONES, totalZONES, NUM_OPTIONS, HORIZON+1))


        for t in range(HORIZON):

            o, pi, xi_all = agent.getOption(cT)


            buffer_nt_z[t] = cT.nt_z
            Rt, Rt_z, cT_new, vioPen, delayPen  = env.step(t, cT, xi_all, pi)
            for z in data.planningZONES:
                for zp in nx.neighbors(data.zGraph, z):
                    t1 = cT_new.nt_z_o[z][zp]
                    if np.sum(cT_new.nt_z_o[z][zp]) > 0 and len(t1[t1 > 0]) > 1:
                        tm1.append(i_episode)
                        for k in range(NUM_OPTIONS):
                            if cT_new.nt_z_o[z][zp][k] > 0:
                                if i_episode not in opt_freq[(z, zp)][k]:
                                    opt_freq[(z, zp)][k][i_episode] = []
                                opt_freq[(z, zp)][k][i_episode].append(cT_new.nt_z_o[z][zp][k])

            buffer_rt_z[t] = Rt_z
            buffer_rt[t] = Rt
            buffer_nt_z_o[t] = cT_new.nt_z_o
            buffer_nt_z_o_tau[t] = cT_new.nt_z_o_tau

            # ------- Other Stats ------ #
            vioCount = np.sum(getVioCount(cT_new.nt_z, data.zGraph))
            trDelay = getTravelDelay(t, cT_new, data.zGraph)
            epRewardList[epCount] = Rt
            epVioCountList[epCount] = vioCount
            epTrDelayList[epCount] = trDelay
            epVioPen[epCount] = vioPen
            epDelayPen[epCount] = delayPen

            epCount += 1
            if i_episode % SHOULD_LOG == 0:
                # -------- Xi
                xiStr = ""
                if t > 0 :
                    # xiStr, xi_avg, pi_avg, xi_avg_all, ent_avg = display_xi_entr_t(o_selected=o, xi_all=xi_all, nt_z=cT.nt_z, xi_avg=xi_avg, pi_avg=pi_avg, xi_avg_all=xi_avg_all, pi=pi, ent_avg=ent_avg)

                    xiStr, xi_avg, pi_avg, xi_avg_all, ent_avg, distStats_avg  = display_xi_entr_t_dist(o_selected=o, xi_all=xi_all, nt_z=cT.nt_z, xi_avg=xi_avg, pi_avg=pi_avg, xi_avg_all=xi_avg_all, pi=pi, ent_avg=ent_avg, dist_stats=distStats_avg)

                ax.writeln("  " + str(t) + " " + str(buffer_rt[t]) + " " + str(buffer_nt_z[t])+" | "+xiStr)
            cT = cT_new

        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))

        epReward = getDiscountedReturn(epRewardList)
        epVioPenDiscount = round(getDiscountedReturn(epVioPen), 3)
        epDelayPenDiscount = round(getDiscountedReturn(epDelayPen), 3)

        buffer_runAvg[sampleID] = epReward
        buffer_VioPenalty[sampleID] = epVioPenDiscount
        buffer_DelayPenalty[sampleID] = epDelayPenDiscount

        buffer_vioCountAvg[sampleID] = np.sum(epVioCountList)
        buffer_trDelayAvg[sampleID] = np.sum(epTrDelayList)

        # ----------- Store Rollouts ------- #
        agent.storeRollouts(buffer_nt_z=buffer_nt_z, buffer_rt=buffer_rt, buffer_nt_z_o=buffer_nt_z_o, buffer_rt_z=buffer_rt_z, buffer_nt_z_o_tau=buffer_nt_z_o_tau)

        sampleID += 1

        # ---------- Train Model --------- #
        if batchID % BATCH_SIZE == 0:
            agent.train()
            batchID = 0

        # -------- Save Model ------- #
        if i_episode % SAVE_MODEL == 0:
            agent.save_model()

        if runTime > TOTAL_RUNTIME:
            break

        # -------- Sample Average ------- #
        if i_episode % SAMPLE_AVG == 0:
            # sampleAvg = np.average(buffer_runAvg)
            # sampleAvg_std = np.std(buffer_runAvg)
            # sampleVioCount = np.average(buffer_vioCountAvg)
            # sampleVioCount_std = np.std(buffer_vioCountAvg)
            # sampleTrDelay = np.average(buffer_trDelayAvg)
            # sampleTrDelay_std = np.std(buffer_trDelayAvg)
            # buffer_runAvg = np.zeros(SAMPLE_AVG)
            # buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
            # buffer_trDelayAvg = np.zeros(SAMPLE_AVG)
            sampleID = 0

        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            epVioCount = np.sum(epVioCountList) # average(epVioCountList)
            epTrDelay = np.sum(epTrDelayList)

            # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
            # pdb.set_trace()
            ax.writeln("\n\n  Total Episode Reward : "+ str(round(epReward, 2)))
            ax.writeln("  Total Episode VioPenalty : " + str(round(epVioPenDiscount, 2)))
            ax.writeln("  Total Episode DelayPenalty : " + str(round(epDelayPenDiscount, 2)))
            ax.writeln("  ")
            ax.writeln("  Total Res. Vio : " + str(round(np.sum(epVioCountList), 2)))
            ax.writeln("  Total Travel Delay : " + str(round(np.sum(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Avg. Res. Vio : " + str(round(np.average(epVioCountList), 2))+" std : "+str(round(np.std(epVioCountList), 2)))
            ax.writeln("  Avg. Travel Delay : " + str(round(np.average(epTrDelayList), 2))+" std : "+str(round(np.std(epTrDelayList), 2)))
            ax.writeln("  ")
            ax.writeln("  Max Res. Vio : " + str(round(np.max(epVioCountList), 2)))
            ax.writeln("  Max Travel Delay : " + str(round(np.max(epTrDelayList), 2)))


            if i_episode % SAMPLE_AVG == 0:
                os.system("cp -r runs/* log/" + dirName + "/plots/")
                ax.writeln("\n  Option Stats:")
                ax.writeln("      Distribution    |    Values")
                xiStr, xi_avg2, opDist, xi_all_avg2, ent_avg2, entStr, statsStr, dist_stats2 = display_avg_ent_dist(pi_avg=pi_avg, xi_avg=xi_avg, ax=ax, xi_avg_all=xi_avg_all, ent_avg=ent_avg, dist_stats=distStats_avg)
                ax.writeln("\n  Average Xi : ")
                ax.writeln("    " + xiStr)
                ax.writeln("\n  Average Entropy : ")
                ax.writeln("    " + entStr)
                agent.log(i_episode, epReward, epReward, epVioCount, epTrDelay, opDist, xi_avg2, xi_all_avg2, ent_avg2)

                ax.writeln("\n  Travel Time Distribution Stats : ")
                ax.writeln("    " + statsStr)
                tstr = "  Mode # "
                for z in data.planningZONES:
                    for zp in nx.neighbors(data.zGraph, z):
                        tstr += str(z) + "_" + str(zp)
                        tstr2 = ""
                        for k in range(NUM_OPTIONS):
                            tstr2 += " : " + str(dist_stats2[(z, zp)][k][1][0]) + "," + str(
                                dist_stats2[(z, zp)][k][1][1])
                        tstr += tstr2 + " | "
                ax.writeln(tstr[:-3])
                opt_freq_ep = {}
                for z in data.planningZONES:
                    for zp in nx.neighbors(data.zGraph, z):
                        opt_freq_ep[(z, zp)] = {}
                        for o in range(NUM_OPTIONS):
                            if i_episode in opt_freq[(z, zp)][o]:
                                t1 = np.array(opt_freq[(z, zp)][o][i_episode])
                                opt_freq_ep[(z, zp)][o] = (round(np.mean(t1),2), round(np.std(t1), 2))
                tstr = "OptFreq # "
                for (z, zp) in opt_freq_ep:
                    tstr += str(z)+"_"+str(zp) +" : "
                    for k in opt_freq_ep[(z, zp)]:
                        tstr += str(opt_freq_ep[(z, zp)][k][0])+","+str(opt_freq_ep[(z, zp)][k][1]) +" : "
                    tstr = tstr[:-3]
                    tstr += " | "
                tstr = tstr[:-3]
                ax.writeln("\n\n  "+tstr)


            ax.writeln("\n  Runtime : "+str(round(runTime, 3))+" Seconds")
        tEnd = ax.now()
        runTime += ax.getRuntime(tStart2, tEnd)


    ax.writeln("\n--------------------------------------------------")
    totalRunTime = ax.getRuntime(tStart, ax.now())
    ax.writeln("Total Runtime : " + str(round(totalRunTime, 3)) + " Seconds")
    ax.writeln("Training Complete !")
    ax.writeln("--------------------------------------------------")
    # print "\n\n"
    # print env.vizData
    ax.dumpDataStr("./dump/vizData_test", env.vizData)

    # ----------- Evaluate
    if EVAL_EPISODES > 0:
        evaluate(data, env, agent)

# =============================================================================== #

if __name__ == '__main__':
    main()
