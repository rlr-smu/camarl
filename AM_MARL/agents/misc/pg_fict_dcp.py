"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 27 Jun 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print("# ============================ START ============================ #")
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import auxLib as ax
import pdb
import rlcompleter
import numpy as np
from parameters import LOAD_MODEL, TINY, SAVE_MODEL, SEED, KEEP_MODELS, MAX_BINARY_LENGTH, LEARNING_RATE, OPTIMIZER, BATCH_SIZE, DISCOUNT, SHOULD_LOG, MAP_ID, VF_NORM, NUM_CORES, HORIZON, WEIGHT_VARIANCE
from numpy import array
import pdb
import rlcompleter
import torch as tc
import torch
import torch.nn as nn
import networkx as nx
from tensorboardX import SummaryWriter

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

# ipDim = self.totalZONES #(TOTAL_VESSEL + 1) * self.totalZONES
# h1Dim = self.totalZONES
# h2Dim = self.totalZONES
# opDim = self.totalZONES

# ============================================================================ #


class actor(nn.Module):

    def __init__(self, totalZONES):
        super(actor, self).__init__()

        tc.manual_seed(SEED)
        tc.set_num_threads(NUM_CORES)
        self.totalZONES = totalZONES
        self.iDim = 2  # < nt(z), nt(z')>
        self.hDim_1 = 2 * self.iDim
        self.hDim_2 = 2 * self.iDim

        self.linear1 = nn.ModuleList()
        self.tanh1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.tanh2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.beta = nn.ModuleList()
        self.sigmoid = nn.ModuleList()

        layerIdx = 0
        for z in range(self.totalZONES):
            for zp in range(self.totalZONES):
                self.linear1.append(nn.Linear(self.iDim, self.hDim_1, bias=True))
                self.linear1[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.linear1[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.tanh1.append(nn.Tanh())
                self.ln1.append(nn.LayerNorm(self.hDim_1))

                self.linear2.append(nn.Linear(self.hDim_1, self.hDim_2, bias=True))
                self.linear2[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.linear2[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.tanh2.append(nn.Tanh())
                self.ln2.append(nn.LayerNorm(self.hDim_2))

                self.beta.append(nn.Linear(self.hDim_2, 1, bias=True))
                self.beta[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.beta[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.sigmoid.append(nn.Sigmoid())
                layerIdx += 1


    def forward(self, x, dtPt):


        nt = tc.tensor(x).float()
        dtPt = nt.shape[0]
        layerIdx = 0
        output = tc.tensor([])


        for z in range(self.totalZONES):
            local_output = []

            for zp in range(self.totalZONES):
                x = []
                x.append(nt[:, z])
                x.append(nt[:, zp])

                # 1st Layer
                x = tc.stack(x, 1)
                x = self.linear1[layerIdx](x)
                x = self.tanh1[layerIdx](x)
                x = self.ln1[layerIdx](x)

                # 2nd Layer
                x = self.linear2[layerIdx](x)
                x = self.tanh2[layerIdx](x)
                x = self.ln2[layerIdx](x)

                # beta
                x = self.beta[layerIdx](x)
                b = self.sigmoid[layerIdx](x)

                local_output.append(b)

                layerIdx += 1
            local_output = tc.stack(local_output, 1)
            output = tc.cat((output, local_output), 1)

        output = tc.reshape(output, (dtPt, self.totalZONES, self.totalZONES))
        return output


class pg_fict_dcp:

    def __init__(self, data=None, load_model=False, dirName="", load_path=""):

        tc.manual_seed(data.seed)
        tc.set_num_threads(NUM_CORES)
        self.totalZONES = data.totalZONES
        self.planningZONES = data.planningZONES
        self.dummyZONES = data.dummyZONES
        self.termZONES = data.termZONES

        self.T_min_max = data.T_min_max
        # self.TOTAL_VESSEL = data.total_vessels
        self.zGraph = data.zGraph

        # self.instance = instance
        # self.actor = actor(data.totalZONES, data.Mask)
        self.actor = actor(data.totalZONES)
        self.dirName = dirName
        if load_model:
            print("-----------------------")
            print("Loading Old Model")
            print("-----------------------")
            self.actor = tc.load("./loadModel" + "/" + load_path + '/model/model.pt')
            self.actor.eval()
        self.optimizer = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.create_train_parameters()
        self.writer = SummaryWriter()
        self.loss = 0

        i = 0
        self.layer_hash = {}
        for z in range(self.totalZONES):
            for zp in range(self.totalZONES):
                self.layer_hash[(z, zp)] = i
                i += 1


    def create_train_parameters(self):

        self.target = []
        self.nt_z_1hot = []
        self.epoch = 1
        self.meanAds = 0
        self.stdAds = 0
        self.ntz_zt = []
        self.Q = []
        self.dataPt = BATCH_SIZE * (HORIZON)
        # tau' - t - Tmin_zz
        self.tau_p_t_tmin = np.zeros((self.dataPt, self.totalZONES, self.totalZONES, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(self.totalZONES):
                    for zp in range(self.totalZONES):
                        tMin_zz = self.T_min_max[z][zp][0]
                        tMax_zz = self.T_min_max[z][zp][1]
                        for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                            if tau_p < HORIZON:
                                self.tau_p_t_tmin[tmpDataPt][z][zp][tau_p] = tau_p - t - tMin_zz
                tmpDataPt += 1
        self.tau_p_t_tmin = tc.tensor(self.tau_p_t_tmin)
        # Tmax_zz - tau' + t
        self.tmax_taup_t = np.zeros((self.dataPt, self.totalZONES, self.totalZONES, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(self.totalZONES):
                    for zp in range(self.totalZONES):
                        tMin_zz = self.T_min_max[z][zp][0]
                        tMax_zz = self.T_min_max[z][zp][1]
                        for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                            if tau_p < HORIZON:
                                self.tmax_taup_t[tmpDataPt][z][zp][tau_p] = tMax_zz - tau_p + t
                tmpDataPt += 1
        self.tmax_taup_t = tc.tensor(self.tmax_taup_t)


    def getBeta(self, cT, i_episode):

        zCount = cT.nt_z
        zCount = np.reshape(zCount, (1, 1 * self.totalZONES))
        beta_t = self.actor(zCount, zCount.shape[0])

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

        return beta_t.data.numpy()

    def clear(self):

        self.Return = []
        self.Q = []
        self.ntz_zt = []
        self.nt_z_1hot = []
        self.target = []

    def train(self, i_episode):

        # y_target = tc.tensor(np.array(self.target)).float()

        x = np.array(self.nt_z_1hot)
        ntzztau = tc.tensor(np.array(self.ntz_zt))
        Adv = tc.tensor(np.array(self.Q))


        dtPt = x.shape[0]
        beta = self.actor(x, x.shape[0])

        # ---- log(xi^t_zz)
        beta_log =  tc.log(tc.add(beta, TINY))
        beta_log = tc.reshape(beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ---- log(1 - xi^t_zz)
        ones = tc.ones((dtPt, self.totalZONES, self.totalZONES))
        one_beta = tc.sub(ones, beta)
        one_beta_log = tc.log(tc.add(one_beta, TINY))
        one_beta_log = tc.reshape(one_beta_log, (dtPt, self.totalZONES, self.totalZONES, 1))

        # ----- nt(z, z', tau_p) * [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))]

        op3 = tc.mul(self.tau_p_t_tmin, beta_log.double())
        op4 = tc.mul(self.tmax_taup_t, one_beta_log.double())
        op5 = tc.add(op3, op4)
        op6 = tc.mul(op5, ntzztau)
        op7 = tc.mul(op6, Adv)


        op8 = tc.sum(op7, 3)
        op8 = op8.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        op8 = op8.transpose_(0, 1)
        op8 = op8.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)

        # op7 = tc.reshape(op7, (BATCH_SIZE*HORIZON, self.totalZONES * self.totalZONES * (HORIZON+1)))
        # op8 = tc.sum(op7, 1)
        # loss = -tc.mean(op8)
        loss = tc.sum(-tc.mean(op8, 2))
        self.loss = float(loss.data)
        self.optimizer.zero_grad()
        loss.backward()

        # for z in self.planningZONES:
        #     for zp in nx.neighbors(self.zGraph, z):
        #         indx = self.layer_hash[(z, zp)]
        #         print "--------------"
        #         print z, zp
        #
        #         print "Layer 1", self.actor.linear1[indx].weight.grad.data, tc.sum(self.actor.linear1[indx].weight.grad).data
        #
        #         print "Layer 2", self.actor.linear2[indx].weight.grad.data, tc.sum(self.actor.linear2[indx].weight.grad).data
        #
        #         print "Beta Layer", self.actor.beta[indx].weight.grad.data,tc.sum(self.actor.beta[indx].weight.grad).data

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()


        self.optimizer.step()
        # print "----------------------------------"
        # print self.actor.linear1[3].weight.grad[3][4]
        # print self.actor.linear2[3].weight.grad[3][4]
        # print "----------------------------------"

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

    def storeRollouts(self, buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt):

        # critic testing
        # nt_zztau = buffer_ntz_zt

        # ------- Real value ------- #
        for t in range(HORIZON):
            zCount_1hot = buffer_nt_z[t]
            zCount_1hot = np.reshape(zCount_1hot, (1, 1 * self.totalZONES))
            self.nt_z_1hot.append(zCount_1hot[0])

        self.ntz_zt.extend(buffer_ntz_zt)


        # -------------------------------- #
        R_z_t_zp_tp = np.zeros((self.totalZONES, HORIZON, self.totalZONES, HORIZON + 1))
        for t in range(HORIZON):
            for z in self.planningZONES:
                tau = t
                for zp in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][zp][0]
                    tmax = self.T_min_max[z][zp][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON+1)):
                        R_z_t_zp_tp[z][tau][zp][tau_p] = np.sum(buffer_rt_z[t:tau_p+1, z])

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

        # ---------- Compute Q & V - Values ----------- #
        # ------- Q ------ #
        q_tmp = np.zeros((self.totalZONES, HORIZON, self.totalZONES, HORIZON + 1))
        self.V = np.zeros((self.totalZONES, HORIZON + 1))
        for tau in range(HORIZON - 1, -1, -1):
            for z in self.planningZONES:
                for z_p in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][z_p][0]
                    tmax = self.T_min_max[z][z_p][1]
                    for tau_p in range(tau + tmin, min(tau + tmax + 1, HORIZON + 1)):
                        q_tmp[z][tau][z_p][tau_p] = R_z_t_zp_tp[z][tau][z_p][tau_p] + DISCOUNT * self.V[z_p][tau_p]
                # ------- V -------- #
                self.V[z][tau] = np.sum(q_tmp[z][tau][:][:] * buffer_ntz_zt[tau][z][:][:]) / np.max(
                    [np.sum(buffer_ntz_zt[tau][z]), TINY])

        buffer_ntz_zt = np.swapaxes(buffer_ntz_zt, 0, 1)
        if self.epoch == 1:
            self.meanAds = np.sum(q_tmp * buffer_ntz_zt) / HORIZON
            self.stdAds = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buffer_ntz_zt) / (HORIZON))

        else:
            self.meanAds1 = np.sum(q_tmp * buffer_ntz_zt) / (HORIZON)
            try:
                self.stdAds1 = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buffer_ntz_zt ) / (HORIZON))
            except RuntimeWarning as e:
                print('error found:', e)
            self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
            self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds
        q_tmp = (q_tmp - self.meanAds)/(self.stdAds + TINY)
        self.V = (self.V - self.meanAds)/(self.stdAds + TINY)


        # ------------ Advantage ------------ #
        Adv = np.zeros((HORIZON, self.totalZONES, self.totalZONES, HORIZON + 1))
        for t in range(HORIZON):
            for z in self.planningZONES:
                for z_p in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][z_p][0]
                    tmax = self.T_min_max[z][z_p][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
                        Adv[t][z][z_p][tau_p] = q_tmp[z][t][z_p][tau_p]
                        Adv[t][z][z_p][tau_p] -= self.V[z][t]

        self.Q.extend(Adv)

    def log(self, i_episode, epReward, betaAvg2, vio, delay):

        self.writer.add_scalar('Total Rewards', epReward, i_episode)
        self.writer.add_scalar('Total ResVio', vio, i_episode)
        self.writer.add_scalar('Total Delay', delay, i_episode)

        # self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
        # self.writer.add_scalar('Loss', self.loss, i_episode)
        for z in range(self.totalZONES):
            if (z not in self.dummyZONES) and (z not in self.termZONES):
                for zp in nx.neighbors(self.zGraph, z):
                    # pdb.set_trace()
                    self.writer.add_scalar("Beta/"+str(z)+"_"+str(zp), betaAvg2[z][zp], i_episode)

    def save_model(self):

        tc.save(self.actor, 'log/'+self.dirName+'/model/model.pt')

    def ep_init(self):
        return

# =============================================================================== #

