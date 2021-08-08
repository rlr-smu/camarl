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
from bluesky import stack, traf, scr, tools
import os
from pprint import pprint
import time
from ipdb import set_trace
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, GRAD_CLIP, NUM_CORES, POLICY_TRAIN, SEED
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
from data import syn_data, real_data, real_real_data, real_real_data_eval
from utils import discounted_return
from auxLib3 import dumpDataStr, loadDataStr
tc.set_num_threads(NUM_CORES)

# =============================== Variables ================================== #
np.random.seed(SEED)
tc.manual_seed(SEED)

# ============================================================================ #

class mean_q(nn.Module):

    def __init__(self, ip_dim, op_dim, num_edges):
        super(mean_q, self).__init__()
        self.iDim = ip_dim
        self.oDim = op_dim
        HIDDEN_DIM_1 = 100
        self.hDim_1 = num_edges * HIDDEN_DIM_1


        # L1
        self.linear1 = nn.Linear(self.iDim, self.hDim_1, bias=True)
        self.act1 = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(self.hDim_1)

        # L2
        self.linear2 = nn.Linear(self.hDim_1, self.hDim_1, bias=True)
        self.act2 = nn.LeakyReLU()
        self.ln2 = nn.LayerNorm(self.hDim_1)

        self.q = nn.Linear(self.hDim_1, self.oDim, bias=True)
        # self.softmax = nn.Softmax(dim=-1)

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

        q = self.q(x)
        # p = self.softmax(q)
        return q

class mtmf_sm_tgt_lg(syn_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(mtmf_sm_tgt_lg, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        self.target_update = 10
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network_list = []
        self.optimizer_list = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")
            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')
            # -------- Actor
            act = tc.load(self.pro_folder + "/load_model" + "/" + "model_actor_"+self.agent +"_"+str(ep) +".pt")
            self.actor = act
            self.actor.eval()

            # -------- Critic
            q = tc.load(self.pro_folder + "/load_model" + "/" + "model_Q_"+self.agent +"_"+str(ep) +".pt")
            self.Q = q
            self.Q.eval()

        else:
            # -------- Actor
            # ip_dim = (self.num_edges * self.num_los * self.num_los) + self.num_edges + 1
            # self.actor = actor(ip_dim, self.num_actions)
            # self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

            # -------- Mean Critic
            ip_dim = (self.num_edges * self.num_los * self.num_los) + self.num_edges
            op_dim = self.num_actions
            self.Q = mean_q(ip_dim, op_dim, self.num_edges)
            self.q_opt = tc.optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)

            self.Q_tgt = mean_q(ip_dim, op_dim, self.num_edges)
            self.Q_tgt.load_state_dict(self.Q.state_dict())
            self.Q_tgt.eval()

        self.action_return = np.zeros(self.num_edges)
        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()
        self.max_return = -self.huge
        self.action_ac = np.zeros(self.max_ac+self.num_env_ac)

    def policy(self, s, k, mean_action):

        x = np.hstack((s, mean_action))
        q = self.Q(x)
        prob = F.softmax(q)
        prob = prob.data.numpy()

        # action = np.random.choice(self.action_space, 1, p=prob)

        action_id = np.random.choice(self.action_id_space, 1, p=prob)[0]
        return action_id

    def update_mean_action(self, state, edge_ac, mean_action_old):

        s = state.reshape(self.num_edges * self.num_los * self.num_los)
        self.edge_ac = edge_ac
        mean_action = np.zeros(self.num_edges)
        for e in range(self.num_edges):
            if len(self.edge_ac[e]) == 0:
                mean_action[e] = mean_action_old[e]
            else:
                ma_avg = []
                for i in self.edge_ac[e]:
                    ma_id = self.policy(s, i, mean_action_old)
                    ma = float(self.action_space[ma_id])
                    ma_avg.append(ma)
                mean_action[e] = np.mean(ma_avg)
        return mean_action.copy()

    def get_action(self, t, state, mean_action, ac_edge):

        x_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        x_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        act_c = np.empty((0, 1))
        nt = state.reshape(self.num_edges*self.num_los*self.num_los)
        ac_count = 0
        for i in range(len(traf.id)):
            id = int(traf.id[i][2:])
            action_id = self.policy(nt, id, mean_action)
            action = self.action_space[action_id]
            t1 = np.hstack((nt, mean_action, id))
            x_a = np.vstack((x_a, t1))
            # self.action_ac[id] = action
            self.action_ac[id] = int(action_id)

            act_c = np.vstack((act_c, action_id))
            e = int(ac_edge[id])
            # if e != -1:
            #     mean_action[e] = int(action)
            # ---- Setting 0 signal for agent's m action
            # because Q(s, a) is implemented as Q(s)
            mean_action[e] = 0
            # Normalizing the count for Critic
            nt_c = nt / self.max_ac
            t2 = np.hstack((nt_c, mean_action))
            x_c = np.vstack((x_c, t2))
            ac_count += 1

        self.buff_x_c = np.vstack((self.buff_x_c, x_c))
        self.buff_x_a = np.vstack((self.buff_x_a, x_a))
        self.buff_ac_act = np.vstack((self.buff_ac_act, act_c))
        self.buff_ac_id[t] = ac_count
        return self.action_ac

    def get_next_state_train(self, t, state, mean_action, ac_edge):

        xp_c = np.empty((0, self.num_edges * self.num_los * self.num_los + self.num_edges))
        xp_a = np.empty((0, self.num_edges * self.num_los * self.num_los + self.num_edges + 1))
        nt = state.reshape(self.num_edges*self.num_los*self.num_los)

        for i in range(len(traf.id)):
            id = int(traf.id[i][2:])
            e = int(ac_edge[id])
            t1 = np.hstack((nt, mean_action, id))
            xp_a = np.vstack((xp_a, t1))
            # if e != -1:
            #     mean_action[e] = int(action)
            # ---- Setting 0 signal for agent's m action
            # because Q(s, a) is implemented as Q(s)
            mean_action[e] = 0
            # Normalizing the count for Critic
            nt_c = nt / self.max_ac
            # set_trace()
            t1 = np.hstack((nt_c, mean_action))
            xp_c = np.vstack((xp_c, t1))

        self.buff_xp_c = np.vstack((self.buff_xp_c, xp_c))
        self.buff_xp_a = np.vstack((self.buff_xp_a, xp_a))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,  buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        buff_reward = buff_rt.sum(1)
        self.buff_rt = []
        for t in range(1, self.horizon+1):
            rt = buff_reward[t - 1]
            t1 = [rt for _ in range(self.buff_ac_id[t])]
            self.buff_rt.extend(t1)
        self.buff_rt = np.array(self.buff_rt)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # --------- Critic Training
        # -- Prediction
        y_pred_all = self.Q(self.buff_x_c)
        act = tc.LongTensor(self.buff_ac_act)
        y_pred = y_pred_all.gather(1, act).squeeze(1)

        # -- Target
        qp = self.Q_tgt(self.buff_xp_c).detach()

        # ---- Get policy prob
        pp = F.softmax(qp)
        v_mtmf = (qp * pp).sum(1).data.numpy()
        y_target = self.buff_rt + DISCOUNT * v_mtmf
        y_target = tc.tensor(y_target).float()
        loss = F.mse_loss(y_pred, y_target)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        self.writer.add_scalar('Loss' + '/total',loss.data.numpy(), ep)
        self.loss = loss.data.numpy()

        # ---------- Actor Training
        # if ep > POLICY_TRAIN:
        #     _, lg_pi = self.actor(self.buff_x_a)
        #     q_sa = y_pred_all.data.numpy()
        #     q_sa = tc.tensor(q_sa).float()
        #     # --- y_target
        #     op1 = lg_pi * q_sa
        #     pi_loss = -tc.mean(op1)
        #     self.actor_opt.zero_grad()
        #     pi_loss.backward()
        #     self.actor_opt.step()

        # ----- Clear buffer
        self.clear_buffer()

        # ----- Update Target Network
        if ep % self.target_update == 0:
            self.Q_tgt.load_state_dict(self.Q.state_dict())

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.batch_id = 0
        self.buff_x_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        self.buff_xp_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        self.buff_x_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        self.buff_xp_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        self.buff_ac_id = np.zeros(self.horizon+1).astype('int32')
        self.buff_ac_act = np.empty((0, 1))
        self.loss = -1

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        # ---- Weight
        if self.local:
            # # Policy
            # self.writer.add_histogram('Policy_Weight/'+"l1", self.actor.linear1.weight, ep)
            # self.writer.add_histogram('Policy_Weight/'+"l2", self.actor.linear2.weight, ep)
            # self.writer.add_histogram('Policy_Weight/'+"pi", self.actor.pi.weight, ep)

            # Critic
            self.writer.add_histogram('Critic_Weight/'+"l1", self.Q.linear1.weight, ep)
            self.writer.add_histogram('Critic_Weight/'+"l2", self.Q.linear2.weight, ep)
            self.writer.add_histogram('Critic_Weight/'+"pi", self.Q.q.weight, ep)

        for e in range(self.num_edges):
            for a in range(self.num_actions):
                # set_trace()
                self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)

    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")

            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)

            # tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor_' + self.agent +  "_"+str(ep) +".pt")

            tc.save(self.Q, self.pro_folder + '/log/' + self.dir_name + '/model/model_Q_' + self.agent +  "_"+str(ep) +".pt")

class mtmf_sm_tgt_real(real_real_data):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(mtmf_sm_tgt_real, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        self.target_update = 10
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network_list = []
        self.optimizer_list = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")
            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')
            # -------- Actor
            act = tc.load(self.pro_folder + "/load_model" + "/" + "model_actor_"+self.agent +"_"+str(ep) +".pt")
            self.actor = act
            self.actor.eval()

            # -------- Critic
            q = tc.load(self.pro_folder + "/load_model" + "/" + "model_Q_"+self.agent +"_"+str(ep) +".pt")
            self.Q = q
            self.Q.eval()

        else:
            # -------- Actor
            # ip_dim = (self.num_edges * self.num_los * self.num_los) + self.num_edges + 1
            # self.actor = actor(ip_dim, self.num_actions)
            # self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

            # -------- Mean Critic
            ip_dim = (self.num_edges * self.num_los * self.num_los) + self.num_edges
            op_dim = self.num_actions
            self.Q = mean_q(ip_dim, op_dim)
            self.q_opt = tc.optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)

            self.Q_tgt = mean_q(ip_dim, op_dim)
            self.Q_tgt.load_state_dict(self.Q.state_dict())
            self.Q_tgt.eval()

        self.action_return = np.zeros(self.num_edges)
        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()
        self.max_return = -self.huge
        self.action_ac = np.zeros(self.max_ac+self.num_env_ac)

    def policy(self, s, k, mean_action):

        x = np.hstack((s, mean_action))
        q = self.Q(x)
        prob = F.softmax(q)
        prob = prob.data.numpy()
        # action = np.random.choice(self.action_space, 1, p=prob)
        action_id = np.random.choice(self.action_id_space, 1, p=prob)[0]
        return action_id

    def update_mean_action(self, state, edge_ac, mean_action_old):

        s = state.reshape(self.num_edges * self.num_los * self.num_los)
        self.edge_ac = edge_ac
        mean_action = np.zeros(self.num_edges)
        for e in range(self.num_edges):
            if len(self.edge_ac[e]) == 0:
                mean_action[e] = mean_action_old[e]
            else:
                ma_avg = []
                for i in self.edge_ac[e]:
                    ma_id = self.policy(s, i, mean_action_old)
                    ma = float(self.action_space[ma_id])
                    ma_avg.append(ma)
                mean_action[e] = np.mean(ma_avg)
        return mean_action.copy()

    def get_action(self, t, state, mean_action, ac_edge):

        x_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        x_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        act_c = np.empty((0, 1))
        nt = state.reshape(self.num_edges*self.num_los*self.num_los)
        ac_count = 0
        for i in range(len(traf.id)):
            id = int(traf.id[i][2:])
            action_id = self.policy(nt, id, mean_action)
            action = self.action_space[action_id]
            t1 = np.hstack((nt, mean_action, id))
            x_a = np.vstack((x_a, t1))
            self.action_ac[id] = action
            act_c = np.vstack((act_c, action_id))
            e = int(ac_edge[id])
            # if e != -1:
            #     mean_action[e] = int(action)
            # ---- Setting 0 signal for agent's m action
            # because Q(s, a) is implemented as Q(s)
            mean_action[e] = 0
            # Normalizing the count for Critic
            nt_c = nt / self.max_ac
            t2 = np.hstack((nt_c, mean_action))
            x_c = np.vstack((x_c, t2))
            ac_count += 1

        self.buff_x_c = np.vstack((self.buff_x_c, x_c))
        self.buff_x_a = np.vstack((self.buff_x_a, x_a))
        self.buff_ac_act = np.vstack((self.buff_ac_act, act_c))
        self.buff_ac_id[t] = ac_count
        return self.action_ac

    def get_next_state_train(self, t, state, mean_action, ac_edge):

        xp_c = np.empty((0, self.num_edges * self.num_los * self.num_los + self.num_edges))
        xp_a = np.empty((0, self.num_edges * self.num_los * self.num_los + self.num_edges + 1))
        nt = state.reshape(self.num_edges*self.num_los*self.num_los)

        for i in range(len(traf.id)):
            id = int(traf.id[i][2:])
            e = int(ac_edge[id])
            t1 = np.hstack((nt, mean_action, id))
            xp_a = np.vstack((xp_a, t1))
            # if e != -1:
            #     mean_action[e] = int(action)
            # ---- Setting 0 signal for agent's m action
            # because Q(s, a) is implemented as Q(s)
            mean_action[e] = 0
            # Normalizing the count for Critic
            nt_c = nt / self.max_ac
            # set_trace()
            t1 = np.hstack((nt_c, mean_action))
            xp_c = np.vstack((xp_c, t1))

        self.buff_xp_c = np.vstack((self.buff_xp_c, xp_c))
        self.buff_xp_a = np.vstack((self.buff_xp_a, xp_a))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,  buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        buff_reward = buff_rt.sum(1)
        self.buff_rt = []
        for t in range(1, self.horizon+1):
            rt = buff_reward[t - 1]
            t1 = [rt for _ in range(self.buff_ac_id[t])]
            self.buff_rt.extend(t1)
        self.buff_rt = np.array(self.buff_rt)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # --------- Critic Training
        # -- Prediction
        y_pred_all = self.Q(self.buff_x_c)
        act = tc.LongTensor(self.buff_ac_act)
        y_pred = y_pred_all.gather(1, act).squeeze(1)

        # -- Target
        qp = self.Q_tgt(self.buff_xp_c).detach()

        # ---- Get policy prob
        pp = F.softmax(qp)
        v_mtmf = (qp * pp).sum(1).data.numpy()
        y_target = self.buff_rt + DISCOUNT * v_mtmf
        y_target = tc.tensor(y_target).float()
        loss = F.mse_loss(y_pred, y_target)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        self.writer.add_scalar('Loss' + '/total',loss.data.numpy(), ep)
        self.loss = loss.data.numpy()

        # ---------- Actor Training
        # if ep > POLICY_TRAIN:
        #     _, lg_pi = self.actor(self.buff_x_a)
        #     q_sa = y_pred_all.data.numpy()
        #     q_sa = tc.tensor(q_sa).float()
        #     # --- y_target
        #     op1 = lg_pi * q_sa
        #     pi_loss = -tc.mean(op1)
        #     self.actor_opt.zero_grad()
        #     pi_loss.backward()
        #     self.actor_opt.step()

        # ----- Clear buffer
        self.clear_buffer()

        # ----- Update Target Network
        if ep % self.target_update == 0:
            self.Q_tgt.load_state_dict(self.Q.state_dict())


    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.batch_id = 0
        self.buff_x_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        self.buff_xp_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        self.buff_x_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        self.buff_xp_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        self.buff_ac_id = np.zeros(self.horizon+1).astype('int32')
        self.buff_ac_act = np.empty((0, 1))
        self.loss = -1

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        # ---- Weight
        if self.local:
            # Policy
            # self.writer.add_histogram('Policy_Weight/'+"l1", self.actor.linear1.weight, ep)
            # self.writer.add_histogram('Policy_Weight/'+"l2", self.actor.linear2.weight, ep)
            # self.writer.add_histogram('Policy_Weight/'+"pi", self.actor.pi.weight, ep)

            # Critic
            self.writer.add_histogram('Critic_Weight/'+"l1", self.Q.linear1.weight, ep)
            self.writer.add_histogram('Critic_Weight/'+"l2", self.Q.linear2.weight, ep)
            self.writer.add_histogram('Critic_Weight/'+"pi", self.Q.q.weight, ep)

        for e in range(self.num_edges):
            for a in range(self.num_actions):
                # set_trace()
                self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)


    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")
            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)
            # tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor_' + self.agent +  "_"+str(ep) +".pt")
            tc.save(self.Q, self.pro_folder + '/log/' + self.dir_name + '/model/model_Q_' + self.agent +  "_"+str(ep) +".pt")

        # Convergence
        tc.save(self.Q, self.pro_folder + '/log/' + self.dir_name + '/model/model_Q_' + self.agent + ".pt")

class mtmf_sm_tgt_real_eval(real_real_data_eval):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(mtmf_sm_tgt_real_eval, self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        tc.manual_seed(self.seed)

        self.target_update = 10
        self.lg = lg
        self.loss_lg = tc.zeros(self.num_edges)
        self.network_list = []
        self.optimizer_list = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")

            # -------- Critic
            q = tc.load(self.pro_folder + "/load_model" + "/" + "model_Q_"+self.agent +".pt")
            self.Q = q
            self.Q.eval()

            # ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')
            # # -------- Critic
            # q = tc.load(self.pro_folder + "/load_model" + "/" + "model_Q_"+self.agent +"_"+str(ep) +".pt")
            # self.Q = q
            # self.Q.eval()

        else:
            # -------- Actor
            # ip_dim = (self.num_edges * self.num_los * self.num_los) + self.num_edges + 1
            # self.actor = actor(ip_dim, self.num_actions)
            # self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

            # -------- Mean Critic
            ip_dim = (self.num_edges * self.num_los * self.num_los) + self.num_edges
            op_dim = self.num_actions
            self.Q = mean_q(ip_dim, op_dim)
            self.q_opt = tc.optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)

            self.Q_tgt = mean_q(ip_dim, op_dim)
            self.Q_tgt.load_state_dict(self.Q.state_dict())
            self.Q_tgt.eval()

        self.action_return = np.zeros(self.num_edges)
        self.action_prob = np.zeros((self.num_edges, self.num_actions))

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.clear_buffer()
        self.max_return = -self.huge
        # self.action_ac = np.zeros(self.max_ac+self.num_env_ac)
        self.action_ac = np.zeros(self.max_ac + 1000 + self.num_env_ac)

    def policy(self, s, k, mean_action):

        x = np.hstack((s, mean_action))
        q = self.Q(x)
        prob = F.softmax(q)
        prob = prob.data.numpy()
        # action = np.random.choice(self.action_space, 1, p=prob)
        action_id = np.random.choice(self.action_id_space, 1, p=prob)[0]
        return action_id

    def update_mean_action(self, state, edge_ac, mean_action_old):

        s = state.reshape(self.num_edges * self.num_los * self.num_los)
        self.edge_ac = edge_ac
        mean_action = np.zeros(self.num_edges)
        for e in range(self.num_edges):
            if len(self.edge_ac[e]) == 0:
                mean_action[e] = mean_action_old[e]
            else:
                ma_avg = []
                for i in self.edge_ac[e]:
                    ma_id = self.policy(s, i, mean_action_old)
                    ma = float(self.action_space[ma_id])
                    ma_avg.append(ma)
                mean_action[e] = np.mean(ma_avg)
        return mean_action.copy()

    def get_action(self, t, state, mean_action, ac_edge):

        x_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        x_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        act_c = np.empty((0, 1))
        nt = state.reshape(self.num_edges*self.num_los*self.num_los)
        ac_count = 0
        for i in range(len(traf.id)):
            id = int(traf.id[i][2:])
            action_id = self.policy(nt, id, mean_action)
            action = self.action_space[action_id]
            t1 = np.hstack((nt, mean_action, id))
            x_a = np.vstack((x_a, t1))
            self.action_ac[id] = action
            act_c = np.vstack((act_c, action_id))
            e = int(ac_edge[id])
            # if e != -1:
            #     mean_action[e] = int(action)
            # ---- Setting 0 signal for agent's m action
            # because Q(s, a) is implemented as Q(s)
            mean_action[e] = 0
            # Normalizing the count for Critic
            nt_c = nt / self.max_ac
            t2 = np.hstack((nt_c, mean_action))
            x_c = np.vstack((x_c, t2))
            ac_count += 1

        self.buff_x_c = np.vstack((self.buff_x_c, x_c))
        self.buff_x_a = np.vstack((self.buff_x_a, x_a))
        self.buff_ac_act = np.vstack((self.buff_ac_act, act_c))
        self.buff_ac_id[t] = ac_count
        return self.action_ac

    def get_next_state_train(self, t, state, mean_action, ac_edge):

        xp_c = np.empty((0, self.num_edges * self.num_los * self.num_los + self.num_edges))
        xp_a = np.empty((0, self.num_edges * self.num_los * self.num_los + self.num_edges + 1))
        nt = state.reshape(self.num_edges*self.num_los*self.num_los)

        for i in range(len(traf.id)):
            id = int(traf.id[i][2:])
            e = int(ac_edge[id])
            t1 = np.hstack((nt, mean_action, id))
            xp_a = np.vstack((xp_a, t1))
            # if e != -1:
            #     mean_action[e] = int(action)
            # ---- Setting 0 signal for agent's m action
            # because Q(s, a) is implemented as Q(s)
            mean_action[e] = 0
            # Normalizing the count for Critic
            nt_c = nt / self.max_ac
            # set_trace()
            t1 = np.hstack((nt_c, mean_action))
            xp_c = np.vstack((xp_c, t1))

        self.buff_xp_c = np.vstack((self.buff_xp_c, xp_c))
        self.buff_xp_a = np.vstack((self.buff_xp_a, xp_a))

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,  buff_act_prob=None):

        st = self.batch_id * self.horizon
        en = st + self.horizon

        # -------- buff_rt
        buff_reward = buff_rt.sum(1)
        self.buff_rt = []
        for t in range(1, self.horizon+1):
            rt = buff_reward[t - 1]
            t1 = [rt for _ in range(self.buff_ac_id[t])]
            self.buff_rt.extend(t1)
        self.buff_rt = np.array(self.buff_rt)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # --------- Critic Training
        # -- Prediction
        y_pred_all = self.Q(self.buff_x_c)
        act = tc.LongTensor(self.buff_ac_act)
        y_pred = y_pred_all.gather(1, act).squeeze(1)

        # -- Target
        qp = self.Q_tgt(self.buff_xp_c).detach()

        # ---- Get policy prob
        pp = F.softmax(qp)
        v_mtmf = (qp * pp).sum(1).data.numpy()
        y_target = self.buff_rt + DISCOUNT * v_mtmf
        y_target = tc.tensor(y_target).float()
        loss = F.mse_loss(y_pred, y_target)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        self.writer.add_scalar('Loss' + '/total',loss.data.numpy(), ep)
        self.loss = loss.data.numpy()

        # ---------- Actor Training
        # if ep > POLICY_TRAIN:
        #     _, lg_pi = self.actor(self.buff_x_a)
        #     q_sa = y_pred_all.data.numpy()
        #     q_sa = tc.tensor(q_sa).float()
        #     # --- y_target
        #     op1 = lg_pi * q_sa
        #     pi_loss = -tc.mean(op1)
        #     self.actor_opt.zero_grad()
        #     pi_loss.backward()
        #     self.actor_opt.step()

        # ----- Clear buffer
        self.clear_buffer()

        # ----- Update Target Network
        if ep % self.target_update == 0:
            self.Q_tgt.load_state_dict(self.Q.state_dict())

    def clear_buffer(self):

        self.data_pt = self.batch_size * self.horizon
        self.batch_id = 0
        self.buff_x_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        self.buff_xp_c = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges))
        self.buff_x_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        self.buff_xp_a = np.empty((0, self.num_edges*self.num_los*self.num_los + self.num_edges + 1))
        self.buff_ac_id = np.zeros(self.horizon+1).astype('int32')
        self.buff_ac_act = np.empty((0, 1))
        self.loss = -1

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        # ---- Weight
        if self.local:
            # Policy
            # self.writer.add_histogram('Policy_Weight/'+"l1", self.actor.linear1.weight, ep)
            # self.writer.add_histogram('Policy_Weight/'+"l2", self.actor.linear2.weight, ep)
            # self.writer.add_histogram('Policy_Weight/'+"pi", self.actor.pi.weight, ep)

            # Critic
            self.writer.add_histogram('Critic_Weight/'+"l1", self.Q.linear1.weight, ep)
            self.writer.add_histogram('Critic_Weight/'+"l2", self.Q.linear2.weight, ep)
            self.writer.add_histogram('Critic_Weight/'+"pi", self.Q.q.weight, ep)

        for e in range(self.num_edges):
            for a in range(self.num_actions):
                # set_trace()
                self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)


    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")
            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)
            # tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor_' + self.agent +  "_"+str(ep) +".pt")
            tc.save(self.Q, self.pro_folder + '/log/' + self.dir_name + '/model/model_Q_' + self.agent +  "_"+str(ep) +".pt")

        # Convergence
        tc.save(self.Q, self.pro_folder + '/log/' + self.dir_name + '/model/model_Q_' + self.agent + ".pt")

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
