"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 10 Mar 2020
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
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, GRAD_CLIP, NUM_CORES
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax
import matplotlib.pyplot as plt
import io
from torch import autograd
from data import syn_data
from utils import discounted_return
from auxLib3 import dumpDataStr, loadDataStr

tc.set_num_threads(NUM_CORES)
# =============================== Variables ================================== #


# ============================================================================ #

class actor(nn.Module):

    def __init__(self, ip_dim, num_action):
        super(actor, self).__init__()
        self.iDim = ip_dim
        HIDDEN_DIM_1 = 100

        self.hDim_1 = HIDDEN_DIM_1

        # L1
        self.linear1 = nn.Linear(self.iDim, self.hDim_1, bias=True)
        self.act1 = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(self.hDim_1)

        # L2
        self.linear2 = nn.Linear(self.hDim_1, self.hDim_1, bias=True)
        self.act2 = nn.LeakyReLU()
        self.ln2 = nn.LayerNorm(self.hDim_1)

        self.pi = nn.Linear(self.hDim_1, num_action, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.lg_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        x = tc.tensor(x).float()
        # L1
        x = self.linear1(x)
        x = self.act1(x)
        x = self.ln1(x)
        # L2
        x = self.linear2(x)
        x = self.act2(x)
        x = self.ln2(x)

        pi_logit = self.pi(x)
        lg_sm = self.lg_softmax(pi_logit)
        return pi_logit, lg_sm

class ind_lrn(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(ind_lrn,  self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.actor_list = []
        self.optimizer_list = []
        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')


            for e in range(self.num_edges):
                nw = tc.load(self.pro_folder+"/load_model"+"/"+"model_"+str(e)+"_"+self.agent+"_"+str(ep) +".pt")
                self.actor_list.append(nw)
                self.actor_list[e].eval()

        else:
            for e in range(self.num_edges):
                ip_dim = self.num_los * self.num_los
                self.actor_list.append(actor(ip_dim, self.num_actions))
                self.optimizer_list.append(tc.optim.Adam(self.actor_list[e].parameters(), lr=LEARNING_RATE))
        self.action_return = np.zeros(self.num_edges)
        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()
        # self.training_parameters()
        self.max_return = -self.huge


    def get_action(self, state=None):
        with tc.no_grad():
            for e in range(self.num_edges):
                pi_logit, _ = self.actor_list[e](state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)

        self.action_prob = np.asarray(self.action_prob).astype('float64')
        return self.action_prob

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None, buff_act_prob=None):

        st = self.batch_id*self.horizon
        en = st + self.horizon

        im_rw = np.swapaxes(buff_rt, 0, 1)

        np.save(self.pro_folder+"/log/"+self.dir_name+"/im_rw_"+self.agent,im_rw)


        # ------- Return
        for e in range(self.num_edges):
            rt_e = buff_rt[:, e]
            tmpReturn = discounted_return(rt_e)
            self.buff_return[e, st:en] = tmpReturn


        # -------- nt
        self.buff_nt[st:en] = buff_nt

        # -------- nt_ev
        self.buff_ntellv[st:en] = buff_ntellv

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        for e in range(self.num_edges):
            state = self.buff_nt[:,e,:,:].reshape(self.data_pt, self.num_los*self.num_los)

            # ------ Fwd Pass
            _, log_pi = self.actor_list[e](state)
            ntev_e = self.buff_ntellv[:,e,:,:,:]
            ntev = tc.tensor(ntev_e.sum(1).sum(1)).float()
            op1 = tc.mul(ntev, log_pi).sum(1)


            gt = tc.tensor(self.buff_return[e, :]).float()
            op2 = tc.mul(op1, gt)
            pi_loss = -tc.sum(tc.mean(op2))
            self.loss_lg[e] = pi_loss

            # ------- Bwd Pass
            self.optimizer_list[e].zero_grad()
            pi_loss.backward()
            # tc.nn.utils.clip_grad_norm_(self.actor_list[e].parameters(), GRAD_CLIP)
            self.optimizer_list[e].step()

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))

        self.buff_rt = np.zeros(self.data_pt)
        self.buff_return = np.zeros((self.num_edges, self.data_pt))
        self.batch_id = 0

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)


        for e in range(self.num_edges):

            if self.local:
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l1", self.actor_list[e].linear1.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l2", self.actor_list[e].linear2.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_pi", self.actor_list[e].pi.weight, ep)

            # ---- Entropy
            pr = tc.tensor(buff_act_prob[:,e,:])
            entr = Categorical(probs=pr).entropy()
            entr_mean = entr.mean()
            self.writer.add_scalar('Entropy/'+str(e), entr_mean, ep)

    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward

            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")

            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)

            for e in range(self.num_edges):
                tc.save(self.actor_list[e], self.pro_folder+'/log/'+self.dir_name+'/model/model_'+str(e)+"_"+self.agent+"_"+str(ep) +".pt")

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    