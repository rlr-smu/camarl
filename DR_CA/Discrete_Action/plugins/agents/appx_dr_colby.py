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
from data import syn_data, real_real_data
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, NUM_CORES, POLICY_TRAIN, EPOCH, DROPOUT, SHUFFLE
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import pred_plot, discounted_return
import tensorflow as tf
import pdb
import rlcompleter
from bluesky import stack, traf, scr, tools
from auxLib3 import dumpDataStr, loadDataStr
tc.set_num_threads(NUM_CORES)


# ============================================================================ #

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for Reward Fn Approximation for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, rw_dim, size):

        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)

        self.rew_buf = np.zeros(combined_shape(size, rw_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, ntsk, rew):
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.act_buf[self.ptr] = ntsk
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     rew=self.rew_buf[idxs],
                     ntsk=self.act_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class rt_nw(nn.Module):

    def __init__(self, ip_dim):
        super(rt_nw, self).__init__()
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

        # rt
        self.rt = nn.Linear(self.hDim_1, 1, bias=True)


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
        rt = self.rt(x)
        return rt

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

class appx_dr_colby(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None, agent_name=None):
        super(appx_dr_colby, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)
        self.agent_name = agent_name

        # -------------------- #
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        # self.rt_nw_list = []
        # self.rt_nw_opt = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            ep = loadDataStr(self.pro_folder + '/load_model/' + self.agent + '_max_ep')

            # Reward Network
            # --------------
            self.rt_nw = tc.load(
                self.pro_folder + '/load_model' + '/model_rt' + "_" + self.agent + "_" + str(ep) + ".pt")
            self.rt_nw.eval()

            # Policy Network
            # --------------
            self.actor_list = []
            for e in range(self.num_edges):
                self.actor_list.append(tc.load(
                    self.pro_folder + '/load_model' + '/model_' + str(e) + "_" + self.agent + "_" + str(ep) + ".pt"))
                self.actor_list[e].eval()


        else:
            self.rw_nw_list = []
            self.rw_opt_list = []
            self.actor_list = []
            self.act_opt_list = []

            for i in range(self.max_ac):
                # Actor Network
                local_state_dim = 3
                ip_dim =  local_state_dim + self.num_los * self.num_los
                self.actor_list.append(actor(ip_dim, self.num_actions))
                self.act_opt_list.append(tc.optim.Adam(self.actor_list[i].parameters(), lr=LEARNING_RATE))

                # Reward Network
                ip_dim = local_state_dim + 1 + (self.num_los * self.num_los)
                self.rw_nw_list.append(rt_nw(ip_dim))
                self.rw_opt_list.append(tc.optim.Adam(self.rw_nw_list[i].parameters(), lr=LEARNING_RATE))


        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()

        # -------- Others
        self.rt_loss = []
        self.max_return = -self.huge

    def get_action(self, state=None, local_state=None):

        action_ac = {}
        for i in range(len(traf.id)):
            ac_id = int(traf.id[i][2:])
            if ac_id not in local_state:
                x = np.zeros(self.num_los*self.num_los+3)
            else:
                # e = int(local_state[ac_id][0])
                # t1 = state[e].reshape(self.num_los*self.num_los)
                # t2 = local_state[ac_id][:-2]
                # x = np.hstack((t1, t2))
                x = local_state[ac_id]['pi'][3:]

            pi_logit, _ = self.actor_list[ac_id](x)
            prob = F.softmax(pi_logit).data.numpy()
            action_id = np.random.choice(self.action_id_space, 1, p=prob)[0]
            action_ac[ac_id] = action_id
        return action_ac

        # with tc.no_grad():
        #     for e in range(self.num_edges):
        #         pi_logit, _ = self.actor_list[e](state[e].flatten())
        #         self.action_prob[e] = F.softmax(pi_logit)
        # self.action_prob = np.asarray(self.action_prob).astype('float64')
        # return self.action_prob

    def random_action(self):

        return np.random.uniform(0, 1, size=(self.num_edges, self.num_actions))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None, buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt
        rt_list = buff_rt.sum(-1)
        ds_default = np.zeros((self.horizon, self.num_los*self.num_los))

        for i in self.local_buffer:
            pi_x = np.array(self.local_buffer[i]['pi'])
            if pi_x.shape[0] > 0:
                t_list = pi_x[:, 0].astype('int')
                a_list = pi_x[:, 2].astype('int')
                a_list = np.expand_dims(a_list, axis=1)
                ds_def_1 = ds_default[t_list, :]
                x = np.array(self.local_buffer[i]['dr'])
                ds_count = x[:, -1]
                ds_def_1[:, self.cs_id] = ds_count
                x = x[:, 0:-1]
                x = np.hstack((a_list, x, ds_def_1))
                counterFac = self.rw_nw_list[i](x).data.numpy().squeeze()
                r = rt_list[t_list]
                dr = r - counterFac
                dr_ret = discounted_return(dr)
                self.buff_diff_return[i] = dr_ret

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        rt = self.buff_rt.sum(-1)
        loss = nn.MSELoss()
        loss_list = []
        for i in self.local_buffer:
            x_all = np.array(self.local_buffer[i]['pi'])
            if x_all.shape[0] > 0:
                # Reward Network
                x_r = x_all[:,2:]
                t_list = x_all[:, 0].astype('int')
                y_pred = self.rw_nw_list[i](x_r).squeeze(-1)
                y_target = tc.tensor(rt[t_list]).float()
                loss_i = loss(y_pred, y_target)
                self.rw_opt_list[i].zero_grad()
                loss_i.backward()
                self.rw_opt_list[i].step()
                loss_list.append(loss_i.data.numpy())

                # Policy Network
                x_pi = x_all[:, 3:]
                pi_logit, log_pi = self.actor_list[i](x_pi)
                action_probs = F.softmax(pi_logit)
                dist = Categorical(action_probs)
                entropy = dist.entropy()
                gt = tc.tensor(self.buff_diff_return[i]).float().unsqueeze(-1)
                op1 = tc.mul(log_pi, gt).sum(-1)
                op2 = tc.add(op1, self.entropy_weight * entropy)
                pi_loss = (tc.mean(op2))
                self.act_opt_list[i].zero_grad()
                pi_loss.backward()
                self.act_opt_list[i].step()
        self.loss = np.mean(loss_list)
        self.writer.add_scalar('Loss' + '/total', self.loss, ep)

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))
        self.buff_ntev_diff = np.zeros((self.data_pt, self.num_edges, self.num_los * self.num_los * self.num_actions))
        self.batch_id = 0
        self.buff_rt = np.zeros((self.data_pt, self.num_edges))
        self.buff_diff_rt = np.zeros((self.num_edges, self.data_pt))
        # self.buff_diff_return = np.zeros((self.num_edges, self.data_pt))
        # self.Q = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))\
        self.loss = -1

        # -- local buffer
        self.local_buffer = {}
        self.buff_diff_return = {}
        for i in range(self.max_ac):
            self.buff_diff_return[i] = []
            self.local_buffer[i] = {}
            self.local_buffer[i]['pi'] = []
            self.local_buffer[i]['dr'] = []

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):


        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)
        for e in range(self.num_edges):

            if self.local:
                # --------- Reward Nw Weights
                self.writer.add_histogram('Rw_Weight/'+"l1", self.rw_nw_list[e].linear1.weight, ep)
                self.writer.add_histogram('Rw_Weight/'+"l2", self.rw_nw_list[e].linear2.weight, ep)
                self.writer.add_histogram('Rw_Weight/'+"op", self.rw_nw_list[e].rt.weight, ep)

                # -------- Policy Nw Weights
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l1", self.actor_list[e].linear1.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l2", self.actor_list[e].linear2.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_pi", self.actor_list[e].pi.weight, ep)

            # ---- Entropy
            pr = tc.tensor(buff_act_prob[:, e, :])
            entr = Categorical(probs=pr).entropy()
            entr_mean = entr.mean()
            self.writer.add_scalar('Entropy/' + str(e), entr_mean, ep)

            for a in range(self.num_actions):
                # set_trace()
                self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)

    def save_model(self, tot_reward, ep):
        return
        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd + "*.*")

            dumpDataStr(self.pro_folder + '/log/' + self.dir_name + '/model/' + self.agent + '_max_ep', ep)

            # Reward Network
            tc.save(self.rt_nw,
                    self.pro_folder + '/log/' + self.dir_name + '/model/model_rt' + "_" + self.agent + "_" + str(
                        ep) + ".pt")

            # Policy Network
            for e in range(self.num_edges):
                tc.save(self.actor_list[e],
                        self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(
                            e) + "_" + self.agent + "_" + str(ep) + ".pt")

    def update_buffer(self, local_buffer):

        for i in local_buffer:
            self.local_buffer[i]['pi'].append(local_buffer[i]['pi'])
            self.local_buffer[i]['dr'].append(local_buffer[i]['dr'])

class appx_dr_colby_real(real_real_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None, agent_name=None):
        super(appx_dr_colby_real, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)
        self.agent_name = agent_name

        # -------------------- #
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        # self.rt_nw_list = []
        # self.rt_nw_opt = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            ep = loadDataStr(self.pro_folder + '/load_model/' + self.agent + '_max_ep')

            # Reward Network
            # --------------
            self.rt_nw = tc.load(
                self.pro_folder + '/load_model' + '/model_rt' + "_" + self.agent + "_" + str(ep) + ".pt")
            self.rt_nw.eval()

            # Policy Network
            # --------------
            self.actor_list = []
            for e in range(self.num_edges):
                self.actor_list.append(tc.load(
                    self.pro_folder + '/load_model' + '/model_' + str(e) + "_" + self.agent + "_" + str(ep) + ".pt"))
                self.actor_list[e].eval()


        else:
            self.rw_nw_list = []
            self.rw_opt_list = []
            self.actor_list = []
            self.act_opt_list = []

            for i in range(self.max_ac):
                # Actor Network
                local_state_dim = 3
                ip_dim =  local_state_dim + self.num_los * self.num_los
                self.actor_list.append(actor(ip_dim, self.num_actions))
                self.act_opt_list.append(tc.optim.Adam(self.actor_list[i].parameters(), lr=LEARNING_RATE))

                # Reward Network
                ip_dim = local_state_dim + 1 + (self.num_los * self.num_los)
                self.rw_nw_list.append(rt_nw(ip_dim))
                self.rw_opt_list.append(tc.optim.Adam(self.rw_nw_list[i].parameters(), lr=LEARNING_RATE))


        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()

        # -------- Others
        self.rt_loss = []
        self.max_return = -self.huge

    def get_action(self, state=None, local_state=None):

        action_ac = {}
        for i in range(len(traf.id)):
            ac_id = int(traf.id[i][2:])
            if ac_id not in local_state:
                x = np.zeros(self.num_los*self.num_los+3)
            else:
                # e = int(local_state[ac_id][0])
                # t1 = state[e].reshape(self.num_los*self.num_los)
                # t2 = local_state[ac_id][:-2]
                # x = np.hstack((t1, t2))
                x = local_state[ac_id]['pi'][3:]

            pi_logit, _ = self.actor_list[ac_id](x)
            prob = F.softmax(pi_logit).data.numpy()
            action_id = np.random.choice(self.action_id_space, 1, p=prob)[0]
            action_ac[ac_id] = action_id
        return action_ac

        # with tc.no_grad():
        #     for e in range(self.num_edges):
        #         pi_logit, _ = self.actor_list[e](state[e].flatten())
        #         self.action_prob[e] = F.softmax(pi_logit)
        # self.action_prob = np.asarray(self.action_prob).astype('float64')
        # return self.action_prob

    def random_action(self):

        return np.random.uniform(0, 1, size=(self.num_edges, self.num_actions))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None, buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt
        rt_list = buff_rt.sum(-1)
        ds_default = np.zeros((self.horizon, self.num_los*self.num_los))

        for i in self.local_buffer:
            pi_x = np.array(self.local_buffer[i]['pi'])
            if pi_x.shape[0] > 0:
                t_list = pi_x[:, 0].astype('int')
                a_list = pi_x[:, 2].astype('int')
                a_list = np.expand_dims(a_list, axis=1)
                ds_def_1 = ds_default[t_list, :]
                x = np.array(self.local_buffer[i]['dr'])
                ds_count = x[:, -1]
                ds_def_1[:, self.cs_id] = ds_count
                x = x[:, 0:-1]
                x = np.hstack((a_list, x, ds_def_1))
                counterFac = self.rw_nw_list[i](x).data.numpy().squeeze()
                r = rt_list[t_list]
                dr = r - counterFac
                dr_ret = discounted_return(dr)
                self.buff_diff_return[i] = dr_ret

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        rt = self.buff_rt.sum(-1)
        loss = nn.MSELoss()
        loss_list = []
        for i in self.local_buffer:
            x_all = np.array(self.local_buffer[i]['pi'])
            if x_all.shape[0] > 0:
                # Reward Network
                x_r = x_all[:,2:]
                t_list = x_all[:, 0].astype('int')
                y_pred = self.rw_nw_list[i](x_r).squeeze(-1)
                y_target = tc.tensor(rt[t_list]).float()
                loss_i = loss(y_pred, y_target)
                self.rw_opt_list[i].zero_grad()
                loss_i.backward()
                self.rw_opt_list[i].step()
                loss_list.append(loss_i.data.numpy())

                # Policy Network
                x_pi = x_all[:, 3:]
                pi_logit, log_pi = self.actor_list[i](x_pi)
                action_probs = F.softmax(pi_logit)
                dist = Categorical(action_probs)
                entropy = dist.entropy()
                gt = tc.tensor(self.buff_diff_return[i]).float().unsqueeze(-1)
                op1 = tc.mul(log_pi, gt).sum(-1)
                op2 = tc.add(op1, self.entropy_weight * entropy)
                pi_loss = (tc.mean(op2))
                self.act_opt_list[i].zero_grad()
                pi_loss.backward()
                self.act_opt_list[i].step()
        self.loss = np.mean(loss_list)
        self.writer.add_scalar('Loss' + '/total', self.loss, ep)

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.buff_nt = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los))
        self.buff_ntellv = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))
        self.buff_ntev_diff = np.zeros((self.data_pt, self.num_edges, self.num_los * self.num_los * self.num_actions))
        self.batch_id = 0
        self.buff_rt = np.zeros((self.data_pt, self.num_edges))
        self.buff_diff_rt = np.zeros((self.num_edges, self.data_pt))
        # self.buff_diff_return = np.zeros((self.num_edges, self.data_pt))
        # self.Q = np.zeros((self.data_pt, self.num_edges, self.num_los, self.num_los, self.num_actions))\
        self.loss = -1

        # -- local buffer
        self.local_buffer = {}
        self.buff_diff_return = {}
        for i in range(self.max_ac):
            self.buff_diff_return[i] = []
            self.local_buffer[i] = {}
            self.local_buffer[i]['pi'] = []
            self.local_buffer[i]['dr'] = []

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):


        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)
        for e in range(self.num_edges):

            if self.local:
                # --------- Reward Nw Weights
                self.writer.add_histogram('Rw_Weight/'+"l1", self.rw_nw_list[e].linear1.weight, ep)
                self.writer.add_histogram('Rw_Weight/'+"l2", self.rw_nw_list[e].linear2.weight, ep)
                self.writer.add_histogram('Rw_Weight/'+"op", self.rw_nw_list[e].rt.weight, ep)

                # -------- Policy Nw Weights
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l1", self.actor_list[e].linear1.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_l2", self.actor_list[e].linear2.weight, ep)
                self.writer.add_histogram('Policy_Weight/'+str(e)+"_pi", self.actor_list[e].pi.weight, ep)

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
            os.system(cmd + "*.*")

            dumpDataStr(self.pro_folder + '/log/' + self.dir_name + '/model/' + self.agent + '_max_ep', ep)

            for i in range(self.max_ac):

                # Reward Network
                tc.save(self.rw_nw_list[i], self.pro_folder + '/log/' + self.dir_name + '/model/model_rt_'+ str(i)+ "_" + self.agent + "_" + str(ep) + ".pt")

                # Policy Network
                tc.save(self.actor_list[i], self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(i) + "_" + self.agent + "_" + str(ep) + ".pt")

    def update_buffer(self, local_buffer):

        for i in local_buffer:
            self.local_buffer[i]['pi'].append(local_buffer[i]['pi'])
            self.local_buffer[i]['dr'].append(local_buffer[i]['dr'])

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
