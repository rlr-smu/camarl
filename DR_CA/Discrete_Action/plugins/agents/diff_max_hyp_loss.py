"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 25 Apr 2020
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
from data import syn_data, real_data
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, NUM_CORES, POLICY_TRAIN, EPOCH, DROPOUT, SHUFFLE

import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import pred_plot, discounted_return
import tensorflow as tf
import pdb
import rlcompleter
from auxLib3 import dumpDataStr, loadDataStr

tc.set_num_threads(NUM_CORES)
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# plt.ioff()

# ============================================================================ #

class rt_nw(nn.Module):

    def __init__(self, ip_dim, num_edges, op_dim):
        super(rt_nw, self).__init__()

        self.num_edges = num_edges
        self.ip_dim = ip_dim
        self.h_dim1 = 100
        self.h_dim2 = 100
        self.o_dim = op_dim

        self.drop1 = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.act1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.drop2 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.act2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.drop_op = nn.ModuleList()
        self.op = nn.ModuleList()

        for e in range(self.num_edges):
            # ---- Layer1
            self.linear1.append(nn.Linear(ip_dim, self.h_dim1, bias=True))
            self.act1.append(nn.LeakyReLU())
            self.ln1.append(nn.LayerNorm(self.h_dim1))

            # ---- Layer2
            self.drop2.append(nn.Dropout(p=DROPOUT))
            self.linear2.append(nn.Linear(self.h_dim1, self.h_dim2, bias=True))
            self.act2.append(nn.LeakyReLU())
            self.ln2.append(nn.LayerNorm(self.h_dim2))

            # ---- Output
            self.drop_op.append(nn.Dropout(p=DROPOUT))
            self.op.append(nn.Linear(self.h_dim2, self.o_dim, bias=True))

    def forward(self, input):

        dtpt = input.shape[0]
        tmp_op = tc.tensor([])

        for e in range(self.num_edges):

            # Mask
            x = input[:,e,:]
            # Layer1
            # x = self.drop1[e](x)
            x = self.linear1[e](x)
            x = self.act1[e](x)
            x = self.ln1[e](x)

            # Layer2
            x = self.drop2[e](x)
            x = self.linear2[e](x)
            x = self.act2[e](x)
            x = self.ln2[e](x)

            # Output
            x = self.drop_op[e](x)
            op = self.op[e](x)

            # vstack
            tmp_op = tc.cat((tmp_op, op), 0)

        output = tmp_op.reshape(dtpt, self.num_edges, self.o_dim)
        return output

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

        # Op
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

        # Op
        pi_logit = self.pi(x)
        lg_sm = self.lg_softmax(pi_logit)
        return pi_logit, lg_sm

class diff_max_hyp_loss(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(diff_max_hyp_loss,  self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        # -------------------- #
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        # self.rt_nw_list = []
        # self.rt_nw_opt = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')

            # Reward Network
            # --------------
            self.rt_nw = tc.load(self.pro_folder + '/load_model' + '/model_rt' + "_" + self.agent +"_"+str(ep) +".pt")
            self.rt_nw.eval()

            # Policy Network
            # --------------
            self.network_list = []
            for e in range(self.num_edges):
                self.network_list.append(tc.load(self.pro_folder + '/load_model'+ '/model_' + str(e) + "_" + self.agent +"_"+str(ep) +".pt"))
                self.network_list[e].eval()


        else:
            # Reward Network
            ip_dim = self.num_los * self.num_los * self.num_actions
            op_dim = self.num_los * self.num_los
            self.rt_nw = rt_nw(ip_dim, self.num_edges, op_dim)
            self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=LEARNING_RATE)

            # Policy Network
            self.network_list = []
            self.optimizer_list = []
            for e in range(self.num_edges):
                ip_dim = self.num_los * self.num_los
                self.network_list.append(network(ip_dim, self.num_actions))
                self.optimizer_list.append(tc.optim.Adam(self.network_list[e].parameters(), lr=LEARNING_RATE))

        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()

        # -------- Others
        self.rt_loss = []
        self.max_return = -self.huge

    def get_action(self, state=None):

        # set_trace()
        with tc.no_grad():
            for e in range(self.num_edges):
                pi_logit, _ = self.network_list[e](state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)
        self.action_prob = np.asarray(self.action_prob).astype('float64')
        return self.action_prob
        # return self.random_action()

    def random_action(self):

        return np.random.uniform(0, 1, size=(self.num_edges, self.num_actions))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,  buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_nt
        self.buff_nt[st:en] = buff_nt

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt

        # -------- buff_ntev
        self.buff_ntellv[st:en] = buff_ntellv

        nsa  = buff_ntellv.reshape(self.horizon, self.num_edges, self.num_los*self.num_los*self.num_actions)

        # -------- Input for rw
        self.buff_ntev_diff[st:en] = nsa

        # -------- Diff Reward
        input = tc.FloatTensor(nsa)
        input = input / self.max_ac
        input.requires_grad = True
        op1 = self.rt_nw(input)
        op2 = self.buff_ntellv.sum(4).reshape(self.horizon, self.num_edges, self.num_los*self.num_los)
        op3 = tc.FloatTensor(op2)
        rt_nn = (op1 * op3).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/self.max_ac

        # Default sa grad
        max_point = -1
        def_grad = grad[:, :, max_point].reshape(self.horizon, self.num_edges, 1)
        grad = grad - def_grad
        diff_rw = grad.numpy()
        diff_all = diff_rw * nsa

        d_rt = diff_all.sum(1).sum(1)
        self.buff_diff_return = discounted_return(d_rt)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # ------- Reward Network
        x_tmp = self.buff_ntev_diff.copy()
        x_tmp = x_tmp / self.max_ac
        r_tmp = self.rw_train
        x_final = np.zeros((self.buff_ntev_diff.shape))
        r_final = np.zeros((self.rw_train.shape))
        loss_list = []
        for i in range(EPOCH):
            for e in range(self.num_edges):
                x_tmp2 = x_tmp[:,e,:]
                r = r_tmp[:,e]
                x_r = np.hstack((x_tmp2, r))
                # Experience Replay
                if SHUFFLE:
                    np.random.shuffle(x_r)
                x_final[:,e,:] = x_r[:,0:-r.shape[1]]
                r_final[:,e] = x_r[:,x_tmp2.shape[1]:]
            x = tc.FloatTensor(x_final)
            y_pred = self.rt_nw(x).squeeze()
            y_target = tc.tensor(r_final).float()
            op1 = y_pred - y_target
            op2 = op1 * op1
            op3 = self.buff_ntellv.sum(4)
            op4 = op3.reshape(self.horizon, self.num_edges, self.num_los*self.num_los)
            op5 = tc.tensor(op4).float()
            op6 = op2 * op5
            loss = self.max_ac * op6.sum()
            self.rt_nw_opt.zero_grad()
            loss.backward()
            self.rt_nw_opt.step()

            dtpt = self.horizon * self.num_edges
            loss = loss / dtpt

            loss_tmp = loss.data.numpy()
            loss_list.append(loss_tmp)
        self.loss = np.mean(loss_list)
        self.writer.add_scalar('Loss' + '/total',self.loss, ep)

        # -------- Policy Network
        if ep > POLICY_TRAIN:
            for e in range(self.num_edges):
                state = self.buff_nt[:, e, :, :].reshape(self.data_pt, self.num_los * self.num_los)
                # ------ Fwd Pass
                pi_logit, log_pi = self.network_list[e](state)
                action_probs = F.softmax(pi_logit)
                dist = Categorical(action_probs)
                entropy = dist.entropy()
                ntev_e = self.buff_ntellv[:,e,:,:,:]

                # q_e = self.Q[:, e, :, :, :]
                # op1 = ntev_e * q_e
                # op2 = tc.tensor(op1.sum(1).sum(1)).float()
                # op3 = tc.mul(op2, log_pi).sum(1)
                # op4 = tc.add(op3, self.entropy_weight*entropy)
                # pi_loss = -(tc.mean(op4))

                # ntev = tc.tensor(ntev_e.sum(1).sum(1)).float()
                # op1 = tc.mul(ntev, log_pi).sum(1)
                # diff_gt = tc.tensor(self.buff_diff_return[e]).float()
                # op2 = tc.mul(op1, diff_gt)
                # op3 = tc.add(op2, self.entropy_weight*entropy)
                # pi_loss = -(tc.mean(op3))

                ntev = tc.tensor(ntev_e.sum(1).sum(1)).float()
                op1 = tc.mul(ntev, log_pi).sum(1)
                gt = tc.tensor(self.buff_diff_return).float()
                op2 = tc.mul(op1, gt)
                op4 = tc.add(op2, self.entropy_weight*entropy)
                pi_loss = -(tc.mean(op4))

                self.optimizer_list[e].zero_grad()
                pi_loss.backward()
                self.optimizer_list[e].step()

    def train_BK(self, ep=None):

        # ------- Reward Network
        x_tmp = self.buff_ntev_diff.copy()
        r_tmp = self.buff_rt.copy()
        x_final = np.zeros((self.buff_ntev_diff.shape))
        r_final = np.zeros((self.buff_rt.shape))

        for i in range(EPOCH):
            for e in range(self.num_edges):
                x_tmp2 = x_tmp[:,e,:]
                r = r_tmp[:,e]
                r = np.expand_dims(r, axis=1)
                x_r = np.hstack((x_tmp2, r))
                # Experience Replay
                if SHUFFLE:
                    np.random.shuffle(x_r)
                x_final[:,e,:] = x_r[:,0:-1]
                r_final[:,e] = x_r[:,-1]
            x = tc.FloatTensor(x_final)
            y_pred = self.rt_nw(x).squeeze()
            y_target = tc.tensor(r_final).float()
            loss = F.mse_loss(y_pred, y_target)
            self.rt_nw_opt.zero_grad()
            loss.backward()
            self.rt_nw_opt.step()
        self.writer.add_scalar('Loss' + '/total',loss.data.numpy(), ep)
        self.loss = loss.data.numpy()

        # -------- Policy Network
        if ep > POLICY_TRAIN:
            for e in range(self.num_edges):
                state = self.buff_nt[:, e, :, :].reshape(self.data_pt, self.num_los * self.num_los)
                # ------ Fwd Pass
                pi_logit, log_pi = self.network_list[e](state)
                action_probs = F.softmax(pi_logit)
                dist = Categorical(action_probs)
                entropy = dist.entropy()
                ntev_e = self.buff_ntellv[:,e,:,:,:]
                ntev = tc.tensor(ntev_e.sum(1).sum(1)).float()
                op1 = tc.mul(ntev, log_pi).sum(1)
                diff_gt = tc.tensor(self.buff_diff_return[e]).float()
                op2 = tc.mul(op1, diff_gt)
                op3 = tc.add(op2, self.entropy_weight*entropy)
                pi_loss = -(tc.mean(op3))
                self.optimizer_list[e].zero_grad()
                pi_loss.backward()
                self.optimizer_list[e].step()

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))
        self.buff_ntev_diff = np.zeros((self.data_pt, self.num_edges, self.num_los*self.num_los*self.num_actions))
        self.batch_id = 0
        self.buff_rt = np.zeros((self.data_pt, self.num_edges))
        self.buff_diff_rt = np.zeros((self.num_edges, self.data_pt))
        # self.buff_diff_return = np.zeros((self.num_edges, self.data_pt))
        self.buff_diff_return = np.zeros((self.data_pt))
        self.loss = -1
        # self.Q = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))


    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):

        return

        # # ---- Reward
        # self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)
        #
        # # ---- Metrics
        # self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        # self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        # self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        # self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)
        #
        #
        # # if self.local:
        # #     # --
        #
        #
        # for e in range(self.num_edges):
        #
        #     if self.local:
        #         # --------- Reward Nw Weights
        #         self.writer.add_histogram('Rw_Weight/'+"l1", self.rt_nw.linear1[e].weight, ep)
        #         self.writer.add_histogram('Rw_Weight/'+"l2", self.rt_nw.linear2[e].weight, ep)
        #         self.writer.add_histogram('Rw_Weight/'+"op", self.rt_nw.op[e].weight, ep)
        #
        #         # -------- Policy Nw Weights
        #         self.writer.add_histogram('Policy_Weight/'+str(e)+"_l1", self.network_list[e].linear1.weight, ep)
        #         self.writer.add_histogram('Policy_Weight/'+str(e)+"_l2", self.network_list[e].linear2.weight, ep)
        #         self.writer.add_histogram('Policy_Weight/'+str(e)+"_pi", self.network_list[e].pi.weight, ep)
        #
        #     # ---- Entropy
        #     pr = tc.tensor(buff_act_prob[:, e, :])
        #     entr = Categorical(probs=pr).entropy()
        #     entr_mean = entr.mean()
        #     self.writer.add_scalar('Entropy/' + str(e), entr_mean, ep)
        #     for a in range(self.num_actions):
        #         # set_trace()
        #         self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)


    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")

            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)


            # Reward Network
            tc.save(self.rt_nw, self.pro_folder + '/log/' + self.dir_name + '/model/model_rt' + "_" + self.agent + "_"+str(ep) +".pt")

            # Policy Network
            for e in range(self.num_edges):
                tc.save(self.network_list[e],
                        self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(e) + "_" + self.agent +  "_"+str(ep) +".pt")

class diff_max_real_hyp(real_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(diff_max_real_hyp,  self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        # -------------------- #
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        # self.rt_nw_list = []
        # self.rt_nw_opt = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')

            # Reward Network
            # --------------
            self.rt_nw = tc.load(self.pro_folder + '/load_model' + '/model_rt' + "_" + self.agent +"_"+str(ep) +".pt")
            self.rt_nw.eval()

            # Policy Network
            # --------------
            self.network_list = []
            for e in range(self.num_edges):
                self.network_list.append(tc.load(self.pro_folder + '/load_model'+ '/model_' + str(e) + "_" + self.agent +"_"+str(ep) +".pt"))
                self.network_list[e].eval()


        else:
            # Reward Network
            ip_dim = self.num_los * self.num_los * self.num_actions
            self.rt_nw = rt_nw(ip_dim, self.num_edges)
            self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=LEARNING_RATE)

            # Policy Network
            self.network_list = []
            self.optimizer_list = []
            for e in range(self.num_edges):
                ip_dim = self.num_los * self.num_los
                self.network_list.append(network(ip_dim, self.num_actions))
                self.optimizer_list.append(tc.optim.Adam(self.network_list[e].parameters(), lr=LEARNING_RATE))

        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()

        # -------- Others
        self.rt_loss = []
        self.max_return = -self.huge

    def get_action(self, state=None):

        # set_trace()
        with tc.no_grad():
            for e in range(self.num_edges):
                pi_logit, _ = self.network_list[e](state[e].flatten())
                self.action_prob[e] = F.softmax(pi_logit)
        self.action_prob = np.asarray(self.action_prob).astype('float64')
        return self.action_prob
        # return self.random_action()

    def random_action(self):

        return np.random.uniform(0, 1, size=(self.num_edges, self.num_actions))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,  buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_nt
        self.buff_nt[st:en] = buff_nt

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt

        # -------- buff_ntev
        self.buff_ntellv[st:en] = buff_ntellv

        nsa  = buff_ntellv.reshape(self.horizon, self.num_edges, self.num_los*self.num_los*self.num_actions)

        # -------- Input for rw
        self.buff_ntev_diff[st:en] = nsa

        # -------- Diff Reward
        input = tc.FloatTensor(nsa)
        input.requires_grad = True
        rt_nn = self.rt_nw(input).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/self.max_ac

        # Default sa grad
        def_grad = grad[:, -1, -1].reshape(self.horizon, 1, 1)
        grad = grad - def_grad
        diff_rw = grad.numpy()
        diff_all = diff_rw * nsa
        diff_all = diff_all.sum(axis=2)
        diff_all = np.swapaxes(diff_all, 0, 1)
        # np.save(self.pro_folder+"/log/"+self.dir_name+"/im_rw", diff_all)

        # ------------------- #
        for e in range(self.num_edges):
            # -------- Diff Return
            tmp_buff = diff_all[e]
            diff_return = discounted_return(tmp_buff)
            self.buff_diff_return[e][st:en] = diff_return

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # ------- Reward Network
        x_tmp = self.buff_ntev_diff.copy()
        r_tmp = self.buff_rt.copy()
        x_final = np.zeros((self.buff_ntev_diff.shape))
        r_final = np.zeros((self.buff_rt.shape))

        for i in range(EPOCH):
            for e in range(self.num_edges):
                x_tmp2 = x_tmp[:,e,:]
                r = r_tmp[:,e]
                r = np.expand_dims(r, axis=1)
                x_r = np.hstack((x_tmp2, r))
                # Experience Replay
                if SHUFFLE:
                    np.random.shuffle(x_r)
                x_final[:,e,:] = x_r[:,0:-1]
                r_final[:,e] = x_r[:,-1]
            x = tc.FloatTensor(x_final)
            y_pred = self.rt_nw(x).squeeze()
            y_target = tc.tensor(r_final).float()
            loss = F.mse_loss(y_pred, y_target)
            self.rt_nw_opt.zero_grad()
            loss.backward()
            self.rt_nw_opt.step()
        self.writer.add_scalar('Loss' + '/total',loss.data.numpy(), ep)

        # -------- Policy Network
        if ep > POLICY_TRAIN:
            for e in range(self.num_edges):
                state = self.buff_nt[:, e, :, :].reshape(self.data_pt, self.num_los * self.num_los)
                # ------ Fwd Pass
                pi_logit, log_pi = self.network_list[e](state)
                action_probs = F.softmax(pi_logit)
                dist = Categorical(action_probs)
                entropy = dist.entropy()
                ntev_e = self.buff_ntellv[:,e,:,:,:]
                ntev = tc.tensor(ntev_e.sum(1).sum(1)).float()
                op1 = tc.mul(ntev, log_pi).sum(1)
                diff_gt = tc.tensor(self.buff_diff_return[e]).float()
                op2 = tc.mul(op1, diff_gt)
                op3 = tc.add(op2, self.entropy_weight*entropy)
                pi_loss = -(tc.mean(op3))
                self.optimizer_list[e].zero_grad()
                pi_loss.backward()
                self.optimizer_list[e].step()

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))
        self.buff_ntev_diff = np.zeros((self.data_pt, self.num_edges, self.num_los*self.num_los*self.num_actions))
        self.batch_id = 0
        self.buff_rt = np.zeros((self.data_pt, self.num_edges))
        self.buff_diff_rt = np.zeros((self.num_edges, self.data_pt))
        self.buff_diff_return = np.zeros((self.num_edges, self.data_pt))

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)


        # if self.local:
        #     # --


        for e in range(self.num_edges):

            if self.local:
                # --------- Reward Nw Weights
                self.writer.add_histogram('Rw_Weight/'+"l1", self.rt_nw.linear1[e].weight, ep)
                self.writer.add_histogram('Rw_Weight/'+"l2", self.rt_nw.linear2[e].weight, ep)
                self.writer.add_histogram('Rw_Weight/'+"op", self.rt_nw.op[e].weight, ep)

                # -------- Policy Nw Weights
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l1", self.network_list[e].linear1.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l2", self.network_list[e].linear2.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_pi", self.network_list[e].pi.weight, ep)

            # ---- Entropy
            pr = tc.tensor(buff_act_prob[:, e, :])
            entr = Categorical(probs=pr).entropy()
            entr_mean = entr.mean()
            self.writer.add_scalar('Entropy/' + str(e), entr_mean, ep)

    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")

            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)


            # Reward Network
            tc.save(self.rt_nw, self.pro_folder + '/log/' + self.dir_name + '/model/model_rt' + "_" + self.agent + "_"+str(ep) +".pt")

            # Policy Network
            for e in range(self.num_edges):
                tc.save(self.network_list[e],
                        self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(e) + "_" + self.agent +  "_"+str(ep) +".pt")

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    