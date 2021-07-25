"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 27 Jun 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print "# ============================ START ============================ #"
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import auxLib as ax
import pdb
import rlcompleter
import numpy as np
from data import totalZONES, HORIZON, listHORIZON, planningZONES, totalVESSELS, dummyZONES, termZONES, Mask, planningTermZones, dirName, T_min_max, minTmin, maxTmax
from parameters import LOAD_MODEL, TINY, SAVE_MODEL, SEED, KEEP_MODELS, MAX_BINARY_LENGTH, LEARNING_RATE, OPTIMIZER, BATCH_SIZE, DISCOUNT, SHOULD_LOG, MAP_ID, VF_NORM, NUM_CORES, MODEL_PATH
from numpy import array
import pdb
import rlcompleter
import torch as tc
import torch
import torch.nn as nn
from actor import actor
import networkx as nx
from tensorboardX import SummaryWriter

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

ipDim = totalZONES #(totalVESSELS + 1) * totalZONES
h1Dim = totalZONES
h2Dim = totalZONES
opDim = totalZONES
tc.manual_seed(SEED)
tc.set_num_threads(NUM_CORES)

# ============================================================================ #

class pg_vanilla:

    def __init__(self, instance, load_model):

        self.instance = instance
        self.actor = actor()
        if load_model:
            print "-----------------------"
            print "Loading Old Model"
            print "-----------------------"
            self.actor = tc.load("./"+MODEL_PATH+"/"+dirName+'/model/model.pt')
            # self.actor = tc.load('./tmp/model.pt')
            self.actor.eval()
        self.optimizer = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.create_train_parameters()
        self.writer = SummaryWriter()

    def create_train_parameters(self):

        self.target = []
        self.nt_z_1hot = []
        self.epoch = 1
        self.meanAds = 0
        self.stdAds = 0
        self.ntz_zt = []
        self.Return = []
        self.dataPt = BATCH_SIZE * (HORIZON)
        # tau' - t - Tmin_zz
        self.tau_p_t_tmin = np.zeros((self.dataPt, totalZONES, totalZONES, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(totalZONES):
                    for zp in range(totalZONES):
                        tMin_zz = T_min_max[z][zp][0]
                        tMax_zz = T_min_max[z][zp][1]
                        for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                            if tau_p < HORIZON:
                                self.tau_p_t_tmin[tmpDataPt][z][zp][tau_p] = tau_p - t - tMin_zz
                tmpDataPt += 1
        self.tau_p_t_tmin = tc.tensor(self.tau_p_t_tmin)
        # Tmax_zz - tau' + t
        self.tmax_taup_t = np.zeros((self.dataPt, totalZONES, totalZONES, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(totalZONES):
                    for zp in range(totalZONES):
                        tMin_zz = T_min_max[z][zp][0]
                        tMax_zz = T_min_max[z][zp][1]
                        for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                            if tau_p < HORIZON:
                                self.tmax_taup_t[tmpDataPt][z][zp][tau_p] = tMax_zz - tau_p + t
                tmpDataPt += 1
        self.tmax_taup_t = tc.tensor(self.tmax_taup_t)

    def getBeta(self, cT, i_episode):

        zCount = cT.nt_z
        # zCount = np.array([[10, 0, 0,0], [8, 2, 0, 0]])
        zCount = np.reshape(zCount, (1, 1 * totalZONES))
        beta_t = self.actor(zCount, zCount.shape[0])
        return beta_t.data.numpy()

    def clear(self):

        self.Return = []
        self.ntz_zt = []
        self.nt_z_1hot = []

    def train(self, i_episode):

        # y_target = tc.tensor(np.array(self.target)).float()

        Return = tc.tensor(np.array(self.Return))

        Return -= tc.mean(Return)
        Return /= (tc.std(Return) + TINY)

        x = np.array(self.nt_z_1hot)
        ntzztau = tc.tensor(np.array(self.ntz_zt))


        dtPt = x.shape[0]
        beta = self.actor(x, x.shape[0])

        # ---- log(xi^t_zz)
        beta_log =  tc.log(tc.add(beta, TINY))
        beta_log = tc.reshape(beta_log, (dtPt, totalZONES, totalZONES, 1))

        # ---- log(1 - xi^t_zz)
        ones = tc.ones((dtPt, totalZONES, totalZONES))
        one_beta = tc.sub(ones, beta)
        one_beta_log = tc.log(tc.add(one_beta, TINY))
        one_beta_log = tc.reshape(one_beta_log, (dtPt, totalZONES, totalZONES, 1))

        # ----- nt(z, z', tau_p) * [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))]
        op3 = tc.mul(self.tau_p_t_tmin, beta_log.double())
        op4 = tc.mul(self.tmax_taup_t, one_beta_log.double())
        op5 = tc.add(op3, op4)
        op6 = tc.mul(op5, ntzztau)
        op7 = tc.reshape(op6, (BATCH_SIZE*HORIZON, totalZONES * totalZONES * (HORIZON+1)))
        op8 = tc.sum(op7, 1)
        op9 = tc.mul(op8, Return)
        loss = -tc.mean(op9)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def storeRollouts(self, buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt):

        # ------- Real value ------- #
        for t in range(HORIZON):
            zCount_1hot = buffer_nt_z[t]
            zCount_1hot = np.reshape(zCount_1hot, (1, 1 * totalZONES))
            self.nt_z_1hot.append(zCount_1hot[0])
        self.ntz_zt.extend(buffer_ntz_zt)

        # ------- Emperical Return ------- #
        return_so_far = 0
        tmpReturn = []
        for t in range(len(buffer_rt)-1, -1, -1):
            return_so_far = buffer_rt[t] + DISCOUNT * return_so_far
            tmpReturn.append(return_so_far)
        tmpReturn = tmpReturn[::-1]
        self.Return.extend(tmpReturn)


    def log(self, i_episode, epReward, betaAvg2, sampleAvg):

        self.writer.add_scalar('Total Rewards', epReward, i_episode)
        self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
        for z in range(totalZONES):
            if (z not in dummyZONES) and (z not in termZONES):
                for zp in self.instance.Zones[z].nbr:
                    # pdb.set_trace()
                    self.writer.add_scalar("Beta/"+str(z)+"_"+str(zp), betaAvg2[z][zp], i_episode)

    def save_model(self, i_episode):

        tc.save(self.actor, 'log/'+dirName+'/model/model.pt')
    def ep_init(self):
        return

# =============================================================================== #

