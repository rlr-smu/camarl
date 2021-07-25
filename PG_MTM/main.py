"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ priImports ================================ #

import sys
import os
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


# ================================ secImports ================================ #

import numpy as np
import networkx as nx
from numpy.random import multinomial
from data import ProblemInstance, HORIZON, totalZONES, totalVESSELS, dummyZONES, termZONES, planningZONES, Pzz_mat, map_id, totalResource, dirName, rCap
from parameters import SAMPLE_AVG, TOTAL_RUNTIME, SEED, EPISODES, SHOULD_LOG, BATCH_SIZE, SAVE_MODEL, LOAD_MODEL, RENDER, wDelay, wResource, capRatio, DISCOUNT, succProb, MAP_ID, MODES, ARR_HORIZON, AGENT_NAME
from environment_fict import Maritime, countTable
from mpa.pgFict.pg_fict import pg_fict
from mpa.ddpg.ddpg import ddpg
# from mpa.dqn_Indp.dqn_indp_iNw import dqn_indp_iNw_Agent
from mpa.pgVanilla.pg_vanilla import pg_vanilla
# from mpa.ac_disc.ac_disc import ac_disc_agent
# from mpa.ac_disc.ac_oneHead import ac_oneHead_agent
from mpa.tmin.tmin import tminAgent
from helperLib import getTravelDelay, getVioCount, getResVio, getResVioNoPenalty, getDelay
import csv
# ============================================================================ #

print "# ============================ START ============================ #"

# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
np.random.seed(SEED)

ax.opFile = ""
if LOAD_MODEL:
    ax.opFile = "./log/"+dirName+"/"+dirName+"_Inference"+".txt"
else:
    ax.opFile = "./log/" + dirName + "/" + dirName + ".txt"


# --------------------- Main ---------------------------- #

def init():

    # ------------------------------ #
    # os.system("rm plots/reward.png")
    ax.deleteDir("./log/"+dirName+"/tboard/1")
    ax.deleteDir("./runs")
    if not LOAD_MODEL:
        ax.deleteDir("./log/"+dirName+"/model")
        os.system("mkdir ./log/"+dirName+"/model")

    os.system("rm " + ax.opFile)
    os.system("cp parameters.py ./log/"+dirName+"/")
    os.system("mkdir ./log/"+dirName+"/tboard")
    # ax.createLog()

def initMap(cT, instance):

    nDummy = len(dummyZONES)
    for dz in dummyZONES:
        nbr = nx.neighbors(instance.zGraph, dz)
        if len(nbr) == 1:
            '''cT.nt_zz[dz][nbr[0]][1] to cT.nt_zz[dz][nbr[0]][0] for arrival rate'''
            # cT.nt_zz[dz][nbr[0]][1] = totalVESSELS / nDummy
            # cT.nt_z[dz] = totalVESSELS / nDummy
            cT.nt_zz[dz][nbr[0]][0] = totalVESSELS / nDummy
            cT.nt_z[dz] = totalVESSELS / nDummy

        else:
            print "Nbr of dummyZone > 1"
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
    instance = ProblemInstance()
    env = Maritime(instance)
    # agent = args.agent
    agent = AGENT_NAME
    if agent == "pg_fict":
        mpa = pg_fict(instance, LOAD_MODEL)
    elif agent == "ddpg":
        mpa = ddpg(instance, LOAD_MODEL)
    elif agent == "pg_vanilla":
        mpa = pg_vanilla(instance, LOAD_MODEL)
    elif agent == "tmin":
        mpa = tminAgent(instance, LOAD_MODEL)
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
    ax.writeln("Map : "+str(map_id))
    ax.writeln("HORIZON : " + str(HORIZON))
    ax.writeln("succProb : " + str(succProb))
    ax.writeln("Total Vessel : " + str(totalVESSELS))
    ax.writeln("Total Zones : " + str(totalZONES))
    ax.writeln("Planning Zones : " + str(planningZONES))
    ax.writeln("Start Zones : " + str(dummyZONES))
    ax.writeln("Terminal Zones : " + str(termZONES))
    ax.writeln("Total Resources : " + str(totalResource))
    ax.writeln("Delay Weight : " + str(wDelay))
    ax.writeln("Resource Weight : " + str(wResource))
    ax.writeln("Resource Capacity : " + str(float(capRatio)))
    ax.writeln("Arrival Modes : " + str(MODES))
    ax.writeln("Arrival Horizon : " + str(ARR_HORIZON))
    ax.writeln("------------------------------")

    buffer_runAvg = np.zeros(SAMPLE_AVG)
    buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
    buffer_trDelayAvg = np.zeros(SAMPLE_AVG)
    sampleID = 0
    batchID = 0

    rollTime = 0
    sampling = 0

    for i_episode in range(1, EPISODES+1):
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

        epCount = 0

        # --------- Buffers --------- #
        cT = countTable()
        cT = initMap(cT, instance)

        buffer_nt_z = np.zeros((HORIZON, totalZONES))
        buffer_ntz_zt = np.zeros((HORIZON, totalZONES, totalZONES, HORIZON+1))
        buffer_rt = np.zeros(HORIZON)
        buffer_rt_z = np.zeros((HORIZON, totalZONES))
        buffer_beta = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_ztz = np.zeros((HORIZON, totalZONES, totalZONES))
        buffer_nt_z_new = np.zeros((HORIZON, totalZONES))
        t1_a = ax.now()
        mpa.ep_init()
        for t in range(HORIZON):
            beta = mpa.getBeta(cT, i_episode)

            buffer_nt_z[t] = cT.nt_z
            Rt, Rt_z, cT_new = env.step(t, cT, beta)

            buffer_ntz_zt[t] = cT_new.ntz_zt
            buffer_rt_z[t] = Rt_z
            buffer_rt[t] = Rt
            buffer_nt_ztz[t] = cT_new.nt_ztz
            buffer_beta[t] = beta[0]
            buffer_nt_z_new[t] = cT_new.nt_z


            # ------- Other Stats ------ #
            vioCount = np.sum(getVioCount(cT_new.nt_z, instance.zGraph))
            trDelay = getTravelDelay(t, cT_new, instance.zGraph)

            epRewardList[epCount] = Rt
            epVioCountList[epCount] = vioCount
            epTrDelayList[epCount] = trDelay
            epCount += 1

            if i_episode % SHOULD_LOG == 0:
                # --------- Beta ---------- #
                betaStr = ""
                if t > 0 :
                    for z in range(totalZONES):
                        if (z not in dummyZONES) and (z not in termZONES):
                            for zp in instance.Zones[z].nbr:
                                if cT.nt_z[z] > 0:
                                    beta_avg[(z, zp)].append(round(beta[0][z][zp], 4))
                                betaStr += "b" + str(z) + str(zp) + " : " + str(round(beta[0][z][zp], 4)) + " "
                ax.writeln("  "+str(t)+" "+str(buffer_rt[t])+ " " +str(buffer_nt_z[t])+" | "+betaStr)
            cT = cT_new

        t1_b = ax.now()
        if i_episode % SHOULD_LOG == 0:
            ax.writeln("  " + str(t + 1) + " " + str("     ") + " " + str(cT_new.nt_z))

        t2 = ax.now()
        sampling += ax.getRuntime(t1_a, t1_b)

        # -------- Book Keeping ------- #
        mpa.storeRollouts(buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt)
        batchID += 1
        # ---------- Train Model --------- #
        t3 = ax.now()
        rollTime += ax.getRuntime(t2, t3)
        if batchID % BATCH_SIZE == 0:
            mpa.train(i_episode)
            mpa.clear()
            batchID = 0
            mpa.epoch += 1

        epReward = getDiscountedReturn(epRewardList)
        buffer_runAvg[sampleID] = epReward
        buffer_vioCountAvg[sampleID] = np.sum(epVioCountList)
        buffer_trDelayAvg[sampleID] = np.sum(epTrDelayList)

        sampleID += 1

        # ----------- Save Model ----------- #
        if i_episode % SAVE_MODEL == 0:
            mpa.save_model(i_episode)

        if runTime > TOTAL_RUNTIME:
            break

        # -------- Sample Average ------- #
        if i_episode % SAMPLE_AVG == 0:
            sampleAvg = np.average(buffer_runAvg)
            sampleAvg_std = np.std(buffer_runAvg)
            sampleVioCount = np.average(buffer_vioCountAvg)
            sampleVioCount_std = np.std(buffer_vioCountAvg)
            sampleTrDelay = np.average(buffer_trDelayAvg)
            sampleTrDelay_std = np.std(buffer_trDelayAvg)
            # samplerResVio = np.average(buffer_resVioAvg)
            # samplerResVio_std = np.std(buffer_resVioAvg)
            # sampleResVioNoPen = np.average(buffer_resVioNoPenAvg)
            # sampleResVioNoPen_std = np.std(buffer_resVioNoPenAvg)


            buffer_runAvg = np.zeros(SAMPLE_AVG)
            buffer_vioCountAvg = np.zeros(SAMPLE_AVG)
            buffer_trDelayAvg = np.zeros(SAMPLE_AVG)
            # buffer_resVioAvg = np.zeros(SAMPLE_AVG)
            # buffer_resVioNoPenAvg = np.zeros(SAMPLE_AVG)

            sampleID = 0

        # --------- Logs ------- #
        if i_episode % SHOULD_LOG == 0:
            epVioCount = np.sum(epVioCountList) # average(epVioCountList)
            epTrDelay = np.sum(epTrDelayList)
            # epResVio = np.sum(epResVioList)
            # epResVioNoPen = float(np.sum(epResVioNoPenList))/(len(planningZONES) * (HORIZON-1))

            ax.writeln("\n  Total Reward : "+ str(round(epReward, 2)))
            ax.writeln("  Total Vio Count : " + str(round(epVioCount, 2)))
            ax.writeln("  Total Avg. Travel Delay : " + str(round(epTrDelay, 2)))
            # ax.writeln("  Total Res Vio : " + str(epResVio))
            # ax.writeln("  Total Res Vio No Penalty : " + str(epResVioNoPen))
            ax.writeln("  ")
            if i_episode % SAMPLE_AVG == 0:
                ax.writeln("  "+str(SAMPLE_AVG)+ " Sample Avg. Reward : "+str(round(sampleAvg, 2))+" std : "+str(round(sampleAvg_std, 2)))
                ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Violation Count : " + str(round(sampleVioCount, 2))+" std : "+str(round(sampleVioCount_std, 2)))
                ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Travel Delay : " + str(round(sampleTrDelay,2)) + " std : " + str(round(sampleTrDelay_std, 2)))
                # ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Res Violation : " + str(samplerResVio)+" std : "+str(samplerResVio_std))
                # ax.writeln("  " + str(SAMPLE_AVG) + " Sample Avg. Res Vio No Pen : " + str(sampleResVioNoPen)+" std : "+str(sampleResVioNoPen_std))

            # -------- Beta ------- #
            betaStr = ""
            for z in range(totalZONES):
                if (z not in dummyZONES) and (z not in termZONES):
                    for zp in instance.Zones[z].nbr:
                        if (z, zp) in beta_avg and len(beta_avg[(z, zp)]) > 0:
                            betaStr += "b"+str(z)+str(zp)+" : "+str(round(average(beta_avg[(z, zp)]), 4)) + " "
            ax.writeln("\n  Average Beta : ")
            ax.writeln("    "+betaStr)
            betaAvg2 = np.zeros((totalZONES, totalZONES))
            for z in range(totalZONES):
                if (z not in dummyZONES) and (z not in termZONES):
                    for zp in instance.Zones[z].nbr:
                        if len(beta_avg[(z, zp)]) > 0:
                            betaAvg2[z][zp] = average(beta_avg[(z, zp)])
            if i_episode % SAMPLE_AVG == 0:
                mpa.log(i_episode, epReward, betaAvg2, sampleAvg)
                os.system("cp -r runs/* ./log/"+dirName+"/tboard/")
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
    print "\n\n# ============================  END  ============================ #"
    """
    sbm = ax.seaBplotsMulti_Bar(HORIZON)
    zone_count = updateZoneCount(t, nt[t])
    zCount.append(list(zone_count))
    plot(sbm, t, zone_count)
    sbm.save("plots/count_large")
    ax.joinPNG(ppath+"plots/")
    """

