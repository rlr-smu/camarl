"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 10 Mar 2020
Description : Vanilla PG with same structure as baseline
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
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, GRAD_CLIP, NUM_CORES
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from scipy.special import softmax
import matplotlib.pyplot as plt
import io
from data import syn_data

tc.set_num_threads(NUM_CORES)
# =============================== Variables ================================== #


# ============================================================================ #

class network(nn.Module):

    def __init__(self, ip_dim, num_action):
        super(network, self).__init__()
        self.iDim = ip_dim
        self.hDim_1 = 32
        self.hDim_2 = 256
        self.hDim_3 = 256

        # L1
        self.linear1 = nn.Linear(self.iDim, self.hDim_1, bias=True)
        self.act1 = nn.ReLU()
        self.ln1 = nn.LayerNorm(self.hDim_1)

        # L2
        self.linear2 = nn.Linear(self.hDim_1, self.hDim_2, bias=True)
        self.act2 = nn.ReLU()
        self.ln2 = nn.LayerNorm(self.hDim_2)

        # L3
        self.linear3 = nn.Linear(self.hDim_2, self.hDim_3, bias=True)
        self.act3 = nn.ReLU()
        self.ln3 = nn.LayerNorm(self.hDim_3)

        # Op
        self.pi = nn.Linear(self.hDim_3, num_action, bias=True)
        self.lg_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        # Input
        x = tc.tensor(x).float()

        # L1
        x = self.linear1(x)
        x = self.act1(x)
        x = self.ln1(x)

        # L2
        x = self.linear2(x)
        x = self.act2(x)
        x = self.ln2(x)

        # L3
        x = self.linear3(x)
        x = self.act3(x)
        x = self.ln3(x)

        # op
        pi_logit = self.pi(x)
        lg_sm = self.lg_softmax(pi_logit)

        return pi_logit, lg_sm

class vpg_bl_multi(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(vpg_bl_multi, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        # -------------------- #
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network_list = []
        self.optimizer_list = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")
            for e in range(self.num_edges):
                nw = tc.load(self.pro_folder+"/load_model"+"/"+"model_"+str(e)+"_"+self.agent+".pt")
                self.network_list.append(nw)
                self.network_list[e].eval()
            # self.network = tc.load("./log"+"/"+dirName+'/model/model.pt')
            # self.network = tc.load("model.pt")
            # self.network.eval()
        else:
            for e in range(self.num_edges):
                ip_dim = self.num_los * self.num_los
                self.network_list.append(network(ip_dim, self.num_actions))
                self.optimizer_list.append(tc.optim.Adam(self.network_list[e].parameters(), lr=LEARNING_RATE))
        self.action_return = np.zeros(self.num_edges)
        self.action_prob = np.zeros((self.num_edges, self.num_actions))
        # self.action_prob = tc.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()
        # self.training_parameters()

    def get_action(self, state=None):
        with tc.no_grad():
            for e in range(self.num_edges):
                pi_logit, _ = self.network_list[e](state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)
        return self.action_prob

    def store_rollouts(self, buff_nt=None, buff_ntev=None, buff_rt=None):


        st = self.batch_id*self.horizon
        en = st + self.horizon

        # ------- Return
        return_so_far = 0
        tmpReturn = []
        for t in range(len(buff_rt)-1, -1, -1):
            return_so_far = buff_rt[t] + DISCOUNT * return_so_far
            tmpReturn.append(return_so_far)
        tmpReturn = tmpReturn[::-1]
        self.buff_return[st:en] = tmpReturn

        # -------- nt
        self.buff_nt[st:en] = buff_nt

        # -------- nt_ev
        self.buff_ntev[st:en] = buff_ntev

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        for e in range(self.num_edges):
            state = self.buff_nt[:,e,:,:].reshape(self.data_pt, self.num_los*self.num_los)
            # ------ Fwd Pass
            _, log_pi = self.network_list[e](state)
            # log_pi = tc.log(pi)
            ntev = tc.tensor(self.buff_ntev[:,e,:]).float()
            gt = tc.tensor(self.buff_return).float()
            op1 = tc.mul(ntev, log_pi).sum(1)
            op2 = tc.mul(op1, gt)
            pi_loss = -tc.sum(tc.mean(op2))
            self.loss_lg[e] = pi_loss
            # ------- Bwd Pass
            self.optimizer_list[e].zero_grad()
            pi_loss.backward()
            # tc.nn.utils.clip_grad_norm_(self.network_list[e].parameters(), GRAD_CLIP)
            self.optimizer_list[e].step()


    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntev = np.zeros((self.data_pt, self.num_edges, self.num_actions))
        self.buff_rt = np.zeros(self.data_pt)
        self.buff_return = np.zeros(self.data_pt)
        self.batch_id = 0

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, goal_reached):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        for e in range(self.num_edges):
            # ---- Weight
            mean1 = tc.mean(self.network_list[e].linear1.weight)
            mean2 = tc.mean(self.network_list[e].linear2.weight)
            mean3 = tc.mean(self.network_list[e].pi.weight)

            self.writer.add_scalar('Weights_'+str(e)+"/mean" +"_l1", mean1, ep)
            self.writer.add_scalar('Weights_'+str(e)+"/mean" +"_l2", mean2, ep)
            self.writer.add_scalar('Weights_'+str(e)+"/mean" +"_pi", mean3, ep)

            # ---- Entropy
            pr = tc.tensor(buff_act_prob[:,e,:])
            entr = Categorical(probs=pr).entropy()
            entr_mean = entr.mean()
            self.writer.add_scalar('Entropy/'+str(e), entr_mean, ep)

    def save_model(self):

        for e in range(self.num_edges):
            tc.save(self.network_list[e], self.pro_folder+'/log/'+self.dir_name+'/model/model_'+str(e)+'.pt')

class vpg_bl_single(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(vpg_bl_single,  self).__init__(dir_name=dir_name, pro_folder=pro_folder)


        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        # --------------------- #
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network = None

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")
            nw = tc.load(self.pro_folder+"/load_model"+"/"+"model"+"_"+self.agent+".pt")
            self.network = nw
            self.network.eval()
        else:
            ip_dim = self.num_los * self.num_los
            self.network = network(ip_dim, self.num_actions)
            self.optimizer = tc.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        self.action_return = np.zeros(self.num_edges)
        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntev = np.zeros((self.data_pt, self.num_edges, self.num_actions))
        self.buff_rt = np.zeros(self.data_pt)
        self.buff_return = np.zeros(self.data_pt)
        self.batch_id = 0

    def get_action(self, state=None):
        with tc.no_grad():
            for e in range(self.num_edges):
                pi_logit, _ = self.network(state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)
        return self.action_prob

    def store_rollouts(self, buff_nt=None, buff_ntev=None, buff_rt=None):

        st = self.batch_id*self.horizon
        en = st + self.horizon

        # ------- Return
        return_so_far = 0
        tmpReturn = []
        for t in range(len(buff_rt)-1, -1, -1):
            return_so_far = buff_rt[t] + DISCOUNT * return_so_far
            tmpReturn.append(return_so_far)
        tmpReturn = tmpReturn[::-1]
        self.buff_return[st:en] = tmpReturn

        # -------- nt
        self.buff_nt[st:en] = buff_nt

        # -------- nt_ev
        self.buff_ntev[st:en] = buff_ntev

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        state = self.buff_nt.reshape(self.data_pt * self.num_edges, self.num_los * self.num_los)
        self.buff_ntev = self.buff_ntev.reshape(self.data_pt * self.num_edges, self.num_actions)

        # ------ Fwd Pass
        _, log_pi = self.network(state)
        ntev = tc.tensor(self.buff_ntev).float()
        op1 = tc.mul(ntev, log_pi).sum(1)

        gt = tc.tensor(self.buff_return).float()
        gt = gt.repeat(self.num_edges)
        op2 = tc.mul(op1, gt)
        pi_loss = -tc.sum(tc.mean(op2))
        self.loss_lg = pi_loss
        # ------- Bwd Pass
        self.optimizer.zero_grad()
        pi_loss.backward()
        # tc.nn.utils.clip_grad_norm_(self.network_list[e].parameters(), GRAD_CLIP)
        self.optimizer.step()

    def train_BK(self):

        with autograd.detect_anomaly():
            for e in range(self.num_edges):
                state = self.buff_nt[:,e,:,:].reshape(self.data_pt, self.num_los*self.num_los)
                # ------ Fwd Pass
                _, log_pi = self.network_list[e](state)
                # log_pi = tc.log(pi)
                ntev = tc.tensor(self.buff_ntev[:,e,:]).float()
                gt = tc.tensor(self.buff_return).float()
                op1 = tc.mul(ntev, log_pi).sum(1)
                op2 = tc.mul(op1, gt)
                pi_loss = -tc.sum(tc.mean(op2))
                self.loss_lg[e] = pi_loss
                # ------- Bwd Pass
                self.optimizer_list[e].zero_grad()
                pi_loss.backward()
                # tc.nn.utils.clip_grad_norm_(self.network_list[e].parameters(), GRAD_CLIP)
                self.optimizer_list[e].step()

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, goal_reached):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        # ---- Weight
        mean1 = tc.mean(self.network.linear1.weight)
        mean2 = tc.mean(self.network.linear2.weight)
        mean3 = tc.mean(self.network.linear3.weight)
        mean4 = tc.mean(self.network.pi.weight)

        self.writer.add_scalar('Weights_'+"/mean" +"_l1", mean1, ep)
        self.writer.add_scalar('Weights_'+"/mean" +"_l2", mean2, ep)
        self.writer.add_scalar('Weights_'+"/mean" + "_l3", mean3, ep)
        self.writer.add_scalar('Weights_'+"/mean" +"_pi", mean4, ep)

        # ---- Entropy
        pr = tc.tensor(buff_act_prob[:,:,:])
        entr = Categorical(probs=pr).entropy()
        entr_mean = entr.mean()
        self.writer.add_scalar('Entropy/', entr_mean, ep)

    def save_model(self):

        tc.save(self.network, self.pro_folder+'/log/'+self.dir_name+'/model/model'+"_"+self.agent+'.pt')

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    