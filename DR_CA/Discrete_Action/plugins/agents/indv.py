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

from data import syn_data, real_data
from utils import discounted_return
from auxLib3 import dumpDataStr, loadDataStr
tc.set_num_threads(NUM_CORES)

# =============================== Variables ================================== #


# ============================================================================ #

class network(nn.Module):

    def __init__(self, ip_dim, num_action):
        super(network, self).__init__()
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

class indv(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(indv, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network_list = []
        self.optimizer_list = []
        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")
            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')
            for e in range(self.num_edges):
                nw = tc.load(self.pro_folder + "/load_model" + "/" + "model_" + str(e) + "_" + self.agent +"_"+str(ep) +".pt")
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

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()
        # self.training_parameters()
        self.max_return = -self.huge

    def get_action(self, state=None):

        with tc.no_grad():
            for e in range(self.num_edges):

                # print(e, state[e].flatten(), state[e].flatten().sum())
                pi_logit, _ = self.network_list[e](state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)
        return self.action_prob

    def store_rollouts(self, ep, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,  buff_n_np=None, train_data=None):

        # ----------- Individual Value Function
        # print("$$$")
        # print(self.V.sum())
        # print(self.Q.sum())
        rw_at = self.at_rw_mat.reshape(self.num_los, self.num_los)
        # print("!!!!!!!!!!!!!")
        for t in range(self.horizon-2, -1, -1):
            s_list = train_data[t]
            for s in s_list:
                rt = rw_at[s[1], s[2]]
                tsum = 0
                for sp in s_list[s]:
                    ntp = buff_ntellv[t+1][sp]
                    indx_list = list(zip(*np.where(ntp > 0)))
                    vp_list = list(map(lambda x: x[0], indx_list))
                    for vp in vp_list:
                        if ntp.sum() > 0 and buff_ntellv[t][s] > 0:
                            indx = (s[0], s[1], s[2], s[3], sp[0], sp[1], sp[2])
                            t1_sum = buff_n_np[t][indx] / buff_ntellv[t][s]
                            # if t1_sum > 1:
                            #     set_trace()

                            t2_sum = ntp[vp] / ntp.sum()
                            tsum += t1_sum * t2_sum * self.V[t+1][sp][vp]

                            # if t == 22:
                            print("%%%%%",t, tsum, t1_sum, t2_sum, self.V[t+1][sp][vp])


                self.V[t][s] = rt + tsum
                # if ep == 4 and t == 22:
                #     print(t, s, self.V[t][s], rt, tsum)

                self.Q[t][s] = (buff_ntellv[t][s]*self.V[t][s]) / self.max_ac

        if ep == 4:
            exit()
        print("&&&&")
        print("V", self.V.sum())
        print("Q", self.Q.sum())

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt.sum(1)
        tmpReturn = discounted_return(self.buff_rt)
        self.buff_return[st:en] = tmpReturn

        # -------- nt
        self.buff_nt[st:en] = buff_nt

        # -------- nt_ev
        self.buff_ntellv[st:en] = buff_ntellv

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # with autograd.detect_anomaly():
        for e in range(self.num_edges):

            state = self.buff_nt[:, e, :, :].reshape(self.data_pt, self.num_los * self.num_los)
            # ------ Fwd Pass
            _, log_pi = self.network_list[e](state)
            ntev_e = self.buff_ntellv[:,e,:,:,:]
            Q_e = self.Q[:,e,:,:,:]
            op1 = ntev_e * Q_e
            op2 = tc.tensor(op1.sum(1).sum(1)).float()
            op3 = tc.mul(op2, log_pi).sum(1)
            pi_loss = -tc.sum(tc.mean(op3))
            self.loss_lg[e] = pi_loss

            # ------- Bwd Pass
            self.optimizer_list[e].zero_grad()
            pi_loss.backward()
            # print("---------------------------------------")
            # print(e)
            # print(Q_e.sum())
            # print(pi_loss)
            # print(self.network_list[e].linear1.weight.grad.sum())
            # print(self.network_list[e].linear2.weight.grad.sum())
            # print(self.network_list[e].pi.weight.grad.sum())

            # tc.nn.utils.clip_grad_norm_(self.network_list[e].parameters(), GRAD_CLIP)
            self.optimizer_list[e].step()

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))

        self.buff_rt = np.zeros(self.data_pt)
        self.buff_return = np.zeros(self.data_pt)
        self.batch_id = 0
        self.loss = -1
        self.Q = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))

        self.V = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))

        rw_at = self.at_rw_mat.reshape(self.num_los, self.num_los)
        for e in range(self.num_edges):
            for l1 in range(self.num_los):
                for l2 in range(self.num_los):
                    for v in range(self.num_actions):
                        self.V[self.horizon-1][e][l1][l2][v] = rw_at[l1][l2]

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        for e in range(self.num_edges):

            # if self.local:
            #     self.writer.add_histogram('Policy_Weight/'+str(e)+"_l1", self.network_list[e].linear1.weight, ep)
            #     self.writer.add_histogram('Policy_Weight/'+str(e)+"_l2", self.network_list[e].linear2.weight, ep)
            #     self.writer.add_histogram('Policy_Weight/'+str(e)+"_pi", self.network_list[e].pi.weight, ep)

            # ---- Entropy
            pr = tc.tensor(buff_act_prob[:, e, :])
            entr = Categorical(probs=pr).entropy()
            entr_mean = entr.mean()
            self.writer.add_scalar('Entropy/' + str(e), entr_mean, ep)
            for a in range(self.num_actions):
                # set_trace()
                self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)


    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")

            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)

            for e in range(self.num_edges):
                tc.save(self.network_list[e],
                        self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(e) + "_" + self.agent +  "_"+str(ep) +".pt")

class vpg_sep_real(real_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(vpg_sep_real, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network_list = []
        self.optimizer_list = []
        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            ep = loadDataStr(self.pro_folder + '/load_model/' + self.agent + '_max_ep')

            for e in range(self.num_edges):
                nw = tc.load(self.pro_folder + "/load_model" + "/" + "model_" + str(e) + "_" + self.agent + "_" + str(
                    ep) + ".pt")
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

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()
        # self.training_parameters()
        self.max_return = -self.huge

    def get_action(self, state=None):

        with tc.no_grad():
            for e in range(self.num_edges):
                pi_logit, _ = self.network_list[e](state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)
        return self.action_prob

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None, buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt.sum(1)
        tmpReturn = discounted_return(self.buff_rt)
        self.buff_return[st:en] = tmpReturn

        # -------- nt
        self.buff_nt[st:en] = buff_nt

        # -------- nt_ev
        self.buff_ntellv[st:en] = buff_ntellv

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # with autograd.detect_anomaly():
        for e in range(self.num_edges):
            state = self.buff_nt[:, e, :, :].reshape(self.data_pt, self.num_los * self.num_los)
            # ------ Fwd Pass
            _, log_pi = self.network_list[e](state)

            ntev_e = self.buff_ntellv[:, e, :, :, :]
            ntev = tc.tensor(ntev_e.sum(1).sum(1)).float()
            op1 = tc.mul(ntev, log_pi).sum(1)

            gt = tc.tensor(self.buff_return).float()
            op2 = tc.mul(op1, gt)
            pi_loss = -tc.sum(tc.mean(op2))
            self.loss_lg[e] = pi_loss

            # ------- Bwd Pass
            self.optimizer_list[e].zero_grad()
            pi_loss.backward()
            # print("---------------------------------------")
            # print(e)
            # print(self.network_list[e].linear1.weight.grad.sum())
            # print(self.network_list[e].linear2.weight.grad.sum())
            # print(self.network_list[e].pi.weight.grad.sum())
            # tc.nn.utils.clip_grad_norm_(self.network_list[e].parameters(), GRAD_CLIP)
            self.optimizer_list[e].step()

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))

        self.buff_rt = np.zeros(self.data_pt)
        self.buff_return = np.zeros(self.data_pt)
        self.batch_id = 0
        self.loss = -1

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
                self.writer.add_histogram('Policy_Weight/' + str(e) + "_l1", self.network_list[e].linear1.weight, ep)
                self.writer.add_histogram('Policy_Weight/' + str(e) + "_l2", self.network_list[e].linear2.weight, ep)
                self.writer.add_histogram('Policy_Weight/' + str(e) + "_pi", self.network_list[e].pi.weight, ep)

            # ---- Entropy
            pr = tc.tensor(buff_act_prob[:, e, :])
            entr = Categorical(probs=pr).entropy()
            entr_mean = entr.mean()
            self.writer.add_scalar('Entropy/' + str(e), entr_mean, ep)

    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd + "*.*")

            dumpDataStr(self.pro_folder + '/log/' + self.dir_name + '/model/' + self.agent + '_max_ep', ep)

            for e in range(self.num_edges):
                tc.save(self.network_list[e],
                        self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(
                            e) + "_" + self.agent + "_" + str(ep) + ".pt")


def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
