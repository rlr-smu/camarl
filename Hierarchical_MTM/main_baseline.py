"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #

import os
import sys

from parameters import SAMPLE_AVG, TOTAL_RUNTIME, SEED, EPISODES, SHOULD_LOG, BATCH_SIZE, SAVE_MODEL, LOAD_MODEL, RENDER, wDelay, wResource, capRatio, DISCOUNT, succProb, MAP_ID, MODES, ARR_HORIZON, TOTAL_VESSEL, HORIZON, EVAL_EPISODES, ENV_VAR, OPENBLAS, OMP, MKL, NUMEXPR

if ENV_VAR:
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS)
    os.environ["OMP_NUM_THREADS"] = str(OMP)
    os.environ["MKL_NUM_THREADS"] = str(MKL)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUMEXPR)

from data import env_data
from environment_bsline import Maritime, countTable
from agents.pgFict_dcp.pg_fict_dcp import pg_fict_dcp
from utils import getTravelDelay, getVioCount
from scipy.stats import binom
import numpy as np
import networkx as nx
import platform
from pprint import pprint
import time
import auxLib as ax
from auxLib import average #, dumpDataStr
import math
import pdb
import rlcompleter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, help='Agent to run')
args = parser.parse_args()

from numpy.random import multinomial

import csv
import math
# ============================================================================ #

print ("# ============================ START ============================ #")

# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
np.random.seed(SEED)
ax.opFile = ""
dirName = str(MAP_ID)+"_"+str(wDelay)+"_"+str(wResource)+"_"+str(capRatio)+"_"+str(HORIZON)+"_"+str(TOTAL_VESSEL)+"_"+args.agent

if LOAD_MODEL:
    ax.opFile = "./log/"+dirName+"/"+dirName+"_Inference"+".txt"
else:
    ax.opFile = "./log/" + dirName + "/" + dirName + ".txt"

# --------------------- Main ---------------------------- #

def init():

    # ------------------------------ #
    # os.system("rm plots/reward.png")
    os.system("mkdir ./log/")
    os.system("mkdir ./log/" + dirName)

    ax.deleteDir("./log/"+dirName+"/plots")
    ax.deleteDir("./runs")
    if not LOAD_MODEL:
        ax.deleteDir("./log/"+dirName+"/model")
        os.system("mkdir ./log/"+dirName+"/model")

    os.system("rm " + ax.opFile)
    os.system("cp parameters.py ./log/"+dirName+"/")
    os.system("mkdir ./log/"+dirName+"/plots")
    os.system("mkdir ./log/" + dirName + "/model")
    # ax.createLog()

def initMap(cT, zGraph, dummyZONES):

    nDummy = len(dummyZONES)
    for dz in dummyZONES:
        nbr = nx.neighbors(zGraph, dz)
        if len(nbr) == 1:
            if TOTAL_VESSEL % nDummy > 0 and dz == 0:
                pop = int(TOTAL_VESSEL / nDummy) + int(TOTAL_VESSEL % nDummy)
            else:
                pop = int(TOTAL_VESSEL / nDummy)
            cT.nt_zz[dz][nbr[0]][0] = pop
            cT.nt_z[dz] = pop
        else:
            print ("Nbr of dummyZone > 1")
            exit()
    return cT

def mapCountTable(cT):

    cT_new = countTable()
    cT_new.nt_z = cT.nt_z
    for z in range(totalZONES):
        cT_new.nt_zt[z][1] = cT.nt_z[z]
    return cT_new

def getDiscountedReturn(tmpRew):

    tmpRew2 = []
    for t in range(len(tmpRew)):
        tmpRew2.append(tmpRew[t]*pow(DISCOUNT, t))
    return sum(tmpRew2)

def main():

    tStart = ax.now()
    runTime = 0
    init()
    data = env_data(mapName=MAP_ID)
    env = Maritime(data=data)
    agent = args.agent
    if agent == "pg_fict_dcp":
        mpa = pg_fict_dcp(load_model=LOAD_MODEL, dirName=dirName, data=data)
    else:
        ax.writeln("Error : Agent not defined!")
        exit()

    tEnd = ax.now()
    runTime += ax.getRuntime(tStart, tEnd)
    ax.writeln(" -------------------------------")
    ax.writeln(" > Compile Time : " + str(round(runTime, 3)) + " Seconds")
    ax.writeln(" -------------------------------")
    ax.writeln(" -------------------------------")
    ax.writeln(" > Training Starts...")
    ax.writeln(" -------------------------------")

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
    ax.writeln("Resource Capacity : " + str(data.rCap))
    ax.writeln("Resource Capacity : " + str(float(capRatio)))
    ax.writeln("Arrival Modes : " + str(MODES))
    ax.writeln("Arrival Horizon : " + str(ARR_HORIZON))
    ax.writeln("------------------------------")

    buffer_runAvg = np.zeros(SAMPLE_AVG)
    buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
    buffer_trDelayAvg = np.zeros(SAMPLE_AVG)

    buffer_VioPenalty = np.zeros(SAMPLE_AVG)
    buffer_DelayPenalty = np.zeros(SAMPLE_AVG)

    sampleID = 0
    batchID = 0

    rollTime = 0
    sampling = 0
    actionTime = 0
    totalZONES = data.totalZONES
    for i_episode in range(1, EPISODES+1):

        # ------- Init Environment
        cT = env.init()

        tStart2 = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("------------------------------")
            ax.writeln("> Episode : "+ str(i_episode))
            beta_avg = {}
            for z in range(totalZONES):
                for zp in range(totalZONES):
                    beta_avg[(z, zp)] = []

        epRewardList = np.zeros(HORIZON+1)
        epVioCountList = np.zeros(HORIZON+1)
        epTrDelayList = np.zeros(HORIZON+1)

        epVioPen = np.zeros(HORIZON+1)
        epDelayPen = np.zeros(HORIZON+1)


        epCount = 0

        # --------- Buffers --------- #
        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        buffer_ntz_zt = np.zeros((HORIZON, totalZONES, totalZONES, HORIZON+1))
        buffer_rt = np.zeros(HORIZON)
        buffer_rt_z = np.zeros((HORIZON, totalZONES))
        buffer_beta = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_ztz = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_z_new = np.zeros((HORIZON, totalZONES))
        t1_a = ax.now()
        mpa.ep_init()

        t1 = ax.now()
        for t in range(HORIZON):

            t1b = ax.now()
            beta = mpa.getBeta(cT, i_episode)
            tmp2 = ax.getRuntime(t1b, ax.now())
            # print "action time", tmp2
            actionTime += tmp2


            buffer_nt_z[t] = cT.nt_z

            t2 = ax.now()
            Rt, Rt_z, cT_new , vioPen, delayPen = env.step(t, cT, beta)
            tmp1 = ax.getRuntime(t2, ax.now())
            sampling += tmp1
            # print "sampling ", tmp1

            buffer_ntz_zt[t] = cT_new.ntz_zt
            buffer_rt_z[t] = Rt_z
            buffer_rt[t] = Rt
            buffer_nt_ztz[t] = cT_new.nt_ztz
            buffer_beta[t] = beta[0]
            buffer_nt_z_new[t] = cT_new.nt_z


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
                # --------- Beta ---------- #
                betaStr = ""
                if t > 0 :
                    for z in range(totalZONES):
                        if (z not in data.dummyZONES) and (z not in data.termZONES):
                            for zp in nx.neighbors(data.zGraph, z):
                                # if cT.nt_z[z] > 0:
                                if buffer_nt_z[t][z] > 0:
                                    beta_avg[(z, zp)].append(round(beta[0][z][zp], 4))
                                betaStr += "b" + str(z) + str(zp) + " : " + str(round(beta[0][z][zp], 4)) + " "
                ax.writeln("  "+str(t)+" "+str(buffer_rt[t])+ " " +str(buffer_nt_z[t])+" | "+betaStr)
            cT = cT_new

        t1_b = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))


        # -------- Book Keeping ------- #
        t3 = ax.now()
        mpa.storeRollouts(buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt)
        # print "Indv Computation", ax.getRuntime(t3, ax.now())
        # print "total sampleing", sampling
        # print "action time", actionTime
        batchID += 1

        # ---------- Train Model --------- #
        if batchID % BATCH_SIZE == 0:
            t4 = ax.now()
            mpa.train(i_episode)
            # print "Train", ax.getRuntime(t4, ax.now())
            mpa.clear()
            batchID = 0
            mpa.epoch += 1

        epReward = getDiscountedReturn(epRewardList)
        epVioPenDiscount = round(getDiscountedReturn(epVioPen), 3)
        epDelayPenDiscount = round(getDiscountedReturn(epDelayPen), 3)

        buffer_runAvg[sampleID] = epReward
        buffer_VioPenalty[sampleID] = epVioPenDiscount
        buffer_DelayPenalty[sampleID] = epDelayPenDiscount

        buffer_vioCountAvg[sampleID] = np.sum(epVioCountList)
        buffer_trDelayAvg[sampleID] = np.sum(epTrDelayList)

        sampleID += 1

        # ----------- Save Model ----------- #
        if i_episode % SAVE_MODEL == 0:
            mpa.save_model()

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
            # samplerResVio = np.average(buffer_resVioAvg)
            # samplerResVio_std = np.std(buffer_resVioAvg)
            # sampleResVioNoPen = np.average(buffer_resVioNoPenAvg)
            # sampleResVioNoPen_std = np.std(buffer_resVioNoPenAvg)
            # buffer_runAvg = np.zeros(SAMPLE_AVG)
            # buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
            # buffer_trDelayAvg = np.zeros(SAMPLE_AVG)
            # buffer_resVioAvg = np.zeros(SAMPLE_AVG)
            # buffer_resVioNoPenAvg = np.zeros(SAMPLE_AVG)

            sampleID = 0

        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            epVioCount = np.sum(epVioCountList) # average(epVioCountList)
            epTrDelay = np.sum(epTrDelayList)
            # epResVio = np.sum(epResVioList)
            # epResVioNoPen = float(np.sum(epResVioNoPenList))/(len(planningZONES) * (HORIZON-1))

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



            # if i_episode % SAMPLE_AVG == 0:
                # ax.writeln("  "+str(SAMPLE_AVG)+ " Sample Avg. Reward : "+str(round(sampleAvg, 2))+" std : "+str(round(sampleAvg_std, 2)))
                # ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Violation Count : " + str(round(sampleVioCount, 2))+" std : "+str(round(sampleVioCount_std, 2)))
                # ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Travel Delay : " + str(round(sampleTrDelay,2)) + " std : " + str(round(sampleTrDelay_std, 2)))
                # ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Res Violation : " + str(samplerResVio)+" std : "+str(samplerResVio_std))
                # ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Res Vio No Pen : " + str(sampleResVioNoPen)+" std : "+str(sampleResVioNoPen_std))

            # -------- Beta ------- #
            betaStr = ""
            for z in range(totalZONES):
                if (z not in data.dummyZONES) and (z not in data.termZONES):
                    for zp in nx.neighbors(data.zGraph, z):
                        if (z, zp) in beta_avg and len(beta_avg[(z, zp)]) > 0:
                            betaStr += "b"+str(z)+str(zp)+" : "+str(round(average(beta_avg[(z, zp)]), 4)) + " "
            ax.writeln("\n  Average Beta : ")
            ax.writeln("    "+betaStr)
            betaAvg2 = np.zeros((totalZONES, totalZONES))
            for z in range(totalZONES):
                if (z not in data.dummyZONES) and (z not in data.termZONES):
                    for zp in nx.neighbors(data.zGraph, z):
                        if len(beta_avg[(z, zp)]) > 0:
                            betaAvg2[z][zp] = average(beta_avg[(z, zp)])
            if i_episode % SAMPLE_AVG == 0:
                mpa.log(i_episode, epReward, betaAvg2, epVioCount, epTrDelay)
                os.system("cp -r runs/* ./log/"+dirName+"/plots/")
            # else:
            #     mpa.tensorboard(i_episode, epReward, betaAvg2, 0)
            ax.writeln("\n  Runtime : "+str(round(runTime, 3))+" Seconds")
        tEnd = ax.now()
        runTime += ax.getRuntime(tStart2, tEnd)


    ax.writeln("\n--------------------------------------------------")
    totalRunTime = ax.getRuntime(tStart, ax.now())
    ax.writeln("Total Runtime : " + str(round(totalRunTime, 3)) + " Seconds")
    ax.writeln("--------------------------------------------------")
    mpa.writer.close()



# =============================================================================== #

if __name__ == '__main__':



    main()
    print ("\n\n# ============================  END  ============================ #")
    """
    sbm = ax.seaBplotsMulti_Bar(HORIZON)
    zone_count = updateZoneCount(t, nt[t])
    zCount.append(list(zone_count))
    plot(sbm, t, zone_count)
    sbm.save("plots/count_large")
    ax.joinPNG(ppath+"plots/")
    """

