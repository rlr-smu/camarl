"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 29 Jun 2018
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
from parameters import LOAD_MODEL, TINY, SAVE_MODEL, SEED, KEEP_MODELS, MAX_BINARY_LENGTH, LEARNING_RATE, OPTIMIZER, BATCH_SIZE, DISCOUNT, SHOULD_LOG, MAP_ID, VF_NORM, EPS_START, EPS_DECAY, EPS_END, NUM_CORES, MODEL_PATH
import networkx as nx
import torch as tc
from model import actor, critic
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

tc.manual_seed(SEED)
np.random.seed(SEED)
tc.set_num_threads(NUM_CORES)
# ============================================================================ #

class ddpg():

    def __init__(self, instance, load_model):

        self.actor = actor()
        self.critic = critic()
        self.instance = instance

        if load_model:
            print "-----------------------"
            print "Loading Old Model"
            print "-----------------------"
            self.actor = tc.load("./"+MODEL_PATH+"/"+dirName+'/model/model_actor.pt')
            self.actor.eval()
            self.critic = tc.load("./"+MODEL_PATH+"/"+dirName+'/model/model_critic.pt')
            self.critic.eval()
            self.eps = EPS_END
        else:
            self.eps = EPS_START

        self.actor_optimizer = tc.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        self.critic_optimizer = tc.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.create_train_parameters()
        self.writer = SummaryWriter()
        self.stepDone = 0

    def create_train_parameters(self):

        self.nt_z_1hot = []
        self.epoch = 1
        self.meanAds = 0
        self.stdAds = 0
        self.ntz_zt = []
        self.nt_ztz = []
        self.buffer_beta = []
        self.Q = []
        self.target = []
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
        zCount = np.reshape(zCount, (1, 1 * totalZONES))
        beta_t = self.actor(zCount, self.eps)
        return beta_t.data.numpy()

    def clear(self):

        self.Return = []
        self.Q = []
        self.ntz_zt = []
        self.nt_z_1hot = []
        self.nt_ztz = []
        self.buffer_beta = []
        self.target = []

    def train(self, i_episode):

        nt = np.array(self.nt_z_1hot)
        nt_ztz = tc.tensor(self.nt_ztz).float()
        beta = tc.tensor(self.buffer_beta).float()

        # if i_episode == 1000:
        #     print "waala"
        #     pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        #     pdb.set_trace()

        # ----------- Train Critic
        y_target = tc.tensor(np.array(self.target)).float()
        y_predicted = self.critic(nt, nt_ztz, beta).float()
        self.loss_critic = []
        self.loss_zz = np.zeros((totalZONES, totalZONES))
        i = 0
        for z in planningZONES:
            for zp in nx.neighbors(self.instance.zGraph, z):
                self.loss_critic.append(F.smooth_l1_loss(y_predicted[:, z, zp], y_target[:, z, zp]))
                self.loss_zz[z][zp] = self.loss_critic[i]
                i += 1
        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

        self.loss_critic = tc.stack(self.loss_critic).sum()
        self.critic_optimizer.zero_grad()
        self.loss_critic.backward()
        self.critic_optimizer.step()

        # ----------- Train Actor
        self.loss_actor = 0
        if i_episode >= 1:
            pred_a1 = self.actor(nt, self.eps)
            loss_actor = -1 * tc.sum(self.critic(nt, nt_ztz, pred_a1))
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.loss_actor = loss_actor.data.numpy()
            # print "Actor Loss : ", self.loss_actor
        # --------- Epsilon Greedy
        self.eps = EPS_END + (EPS_START - EPS_END) * \
                                  math.exp(-1. * self.stepDone / EPS_DECAY)
        self.stepDone += 1

    def storeRollouts(self, buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt):

        nt_zztau = buffer_ntz_zt
        # ------- Real value ------- #
        for t in range(HORIZON):
            zCount_1hot = buffer_nt_z[t]
            zCount_1hot = np.reshape(zCount_1hot, (1, 1 * totalZONES))
            self.nt_z_1hot.append(zCount_1hot[0])

        self.ntz_zt.extend(buffer_ntz_zt)
        self.nt_ztz.extend(buffer_nt_ztz)
        self.buffer_beta.extend(buffer_beta)
        buffer_ntz_zt = buffer_ntz_zt / totalVESSELS

        # -------------------------------- #
        R_z_t_zp_tp = np.zeros((totalZONES, HORIZON, totalZONES, HORIZON + 1))
        for t in range(HORIZON):
            for z in planningZONES:
                tau = t
                for zp in nx.neighbors(self.instance.zGraph, z):
                    tmin = T_min_max[z][zp][0]
                    tmax = T_min_max[z][zp][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
                        R_z_t_zp_tp[z][tau][zp][tau_p] = np.sum(buffer_rt_z[t:tau_p + 1, z])

        # ---------- Compute Q & V - Values ----------- #
        # ------- Q ------ #
        q_tmp = np.zeros((totalZONES, HORIZON, totalZONES, HORIZON + 1))
        self.V = np.zeros((totalZONES, HORIZON + 1))
        for tau in range(HORIZON - 1, -1, -1):
            for z in planningZONES:
                for z_p in nx.neighbors(self.instance.zGraph, z):
                    tmin = T_min_max[z][z_p][0]
                    tmax = T_min_max[z][z_p][1]
                    for tau_p in range(tau + tmin, min(tau + tmax + 1, HORIZON + 1)):
                        q_tmp[z][tau][z_p][tau_p] = R_z_t_zp_tp[z][tau][z_p][tau_p] + DISCOUNT * self.V[z_p][tau_p]
                # ------- V -------- #
                self.V[z][tau] = np.sum(q_tmp[z][tau][:][:] * buffer_ntz_zt[tau][z][:][:]) / np.max(
                    [np.sum(buffer_ntz_zt[tau][z]), TINY])

        buffer_ntz_zt = np.swapaxes(buffer_ntz_zt, 0, 1)
        if self.epoch == 1:
            self.meanAds = np.sum(q_tmp * buffer_ntz_zt) / (len(q_tmp))
            self.stdAds = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buffer_ntz_zt) / (len(q_tmp)))
        else:
            self.meanAds1 = np.sum(q_tmp * buffer_ntz_zt) / (len(q_tmp))
            try:
                self.stdAds1 = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buffer_ntz_zt) / (len(q_tmp)))
            except RuntimeWarning as e:
                print('error found:', e)
            self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
            self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds
        q_tmp = (q_tmp - self.meanAds) / (self.stdAds + TINY)

        # --------- Critic
        target_tmp = np.zeros((HORIZON, totalZONES, totalZONES))
        q_tmp2 = np.swapaxes(q_tmp, 0, 1)
        for t in range(HORIZON):
            for z in planningZONES:
                for zp in nx.neighbors(self.instance.zGraph, z):
                    target_tmp[t][z][zp] = np.sum(nt_zztau[t][z][zp][:]*q_tmp2[t][z][zp][:])

        self.target.extend(target_tmp)

        # Adv = np.swapaxes(q_tmp, 0, 1)
        self.V = (self.V - self.meanAds) / (self.stdAds + TINY)
        # ------------ Advantage ------------ #
        Adv = np.zeros((HORIZON, totalZONES, totalZONES, HORIZON + 1))
        for t in range(HORIZON):
            for z in planningZONES:
                for z_p in nx.neighbors(self.instance.zGraph, z):
                    tmin = T_min_max[z][z_p][0]
                    tmax = T_min_max[z][z_p][1]
                    for tau_p in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
                        Adv[t][z][z_p][tau_p] = q_tmp[z][t][z_p][tau_p]
                        Adv[t][z][z_p][tau_p] -= self.V[z][t]
        self.Q.extend(Adv)

    def log(self, i_episode, epReward, betaAvg2, sampleAvg):

        self.writer.add_scalar('Total Rewards', epReward, i_episode)
        self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
        self.writer.add_scalar('Loss/CriticTotal', self.loss_critic.data.numpy(), i_episode)
        self.writer.add_scalar('Loss/Actor', self.loss_actor, i_episode)
        self.writer.add_scalar('Epsilon', self.eps, i_episode)
        for z in planningZONES:
            for zp in nx.neighbors(self.instance.zGraph, z):
                    self.writer.add_scalar("Loss/" + str(z) + "_" + str(zp), self.loss_zz[z][zp], i_episode)

        for z in range(totalZONES):
            if (z not in dummyZONES) and (z not in termZONES):
                for zp in self.instance.Zones[z].nbr:
                    # pdb.set_trace()
                    self.writer.add_scalar("Beta/" + str(z) + "_" + str(zp), betaAvg2[z][zp], i_episode)

    def save_model(self, i_episode):

        tc.save(self.actor, 'log/' + dirName + '/model/model_actor.pt')
        tc.save(self.critic, 'log/' + dirName + '/model/model_critic.pt')

    def ep_init(self):
        return

# =============================================================================== #


        # nt = np.array([[10, 0, 0,0], [8, 2, 0, 0]])
        # nt_ztz = tc.tensor([[[ 2.,  1.,  2.,  9.],
        #  [ 9.,  4.,  9.,  8.],
        #  [ 4.,  7.,  6.,  2.],
        #  [ 4.,  5.,  9.,  2.]],
        #
        # [[ 5.,  1.,  4.,  3.],
        #  [ 1.,  5.,  3.,  8.],
        #  [ 8.,  9.,  7.,  4.],
        #  [ 8.,  8.,  5.,  6.]]]).float()
        #
        # beta = tc.tensor([[[ 0.8764,  0.8946,  0.0850,  0.0391],
        #  [ 0.1698,  0.8781,  0.0983,  0.4211],
        #  [ 0.9579,  0.5332,  0.6919,  0.3155],
        #  [ 0.6865,  0.8346,  0.0183,  0.7501]],
        #
        # [[ 0.9889,  0.7482,  0.2804,  0.7893],
        #  [ 0.1032,  0.4479,  0.9086,  0.2936],
        #  [ 0.2878,  0.1300,  0.0194,  0.6788],
        #  [ 0.2116,  0.2655,  0.4916,  0.0534]]]).float()



        # nt = np.array([[10, 0, 0, 0]])
        # nt_ztz = tc.tensor([[[ 2.,  1.,  2.,  9.],
        #  [ 9.,  4.,  9.,  8.],
        #  [ 4.,  7.,  6.,  2.],
        #  [ 4.,  5.,  9.,  2.]]]).float()
        #
        # beta = tc.tensor([[[ 0.8764,  0.8946,  0.0850,  0.0391],
        #  [ 0.1698,  0.8781,  0.0983,  0.4211],
        #  [ 0.9579,  0.5332,  0.6919,  0.3155],
        #  [ 0.6865,  0.8346,  0.0183,  0.7501]]]).float()


        # nt = np.array([[8, 2, 0, 0]])
        # nt_ztz = tc.tensor([[[ 5.,  1.,  4.,  3.],
        #  [ 1.,  5.,  3.,  8.],
        #  [ 8.,  9.,  7.,  4.],
        #  [ 8.,  8.,  5.,  6.]]]).float()
        #
        # beta = tc.tensor([[[ 0.9889,  0.7482,  0.2804,  0.7893],
        #  [ 0.1032,  0.4479,  0.9086,  0.2936],
        #  [ 0.2878,  0.1300,  0.0194,  0.6788],
        #  [ 0.2116,  0.2655,  0.4916,  0.0534]]]).float()

