"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 08 Sep 2020
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print("# ============================ START ============================ #")
# ================================ Imports ================================ #
import sys
import os
import ipdb
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, NUM_CORES, POLICY_TRAIN, EPOCH, DROPOUT, SHUFFLE, SEED, AGENT, GRID, NUM_AGENTS, NUM_ACTIONS, HORIZON, BATCH_SIZE, ENTROPY_WEIGHT, LOCAL, HUGE
import numpy as np
from auxLib3 import dumpDataStr, loadDataStr
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from ipdb import set_trace
from utils import discounted_return
# ============================================================================ #

class rt_nw(nn.Module):

    def __init__(self, ip_dim):
        super(rt_nw, self).__init__()
        self.ip_dim = ip_dim
        self.h_dim1 = 100
        self.h_dim2 = 100
        self.o_dim = 1

        # self.drop1 = nn.ModuleList()
        self.linear1 = nn.Linear(ip_dim, self.h_dim1, bias=True)
        self.act1 = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(self.h_dim1)
        self.drop2 = nn.Dropout(p=DROPOUT)
        self.linear2 = nn.Linear(self.h_dim1, self.h_dim2, bias=True)
        self.act2 = nn.LeakyReLU()
        self.ln2 = nn.LayerNorm(self.h_dim2)
        self.drop_op = nn.Dropout(p=DROPOUT)
        self.op = nn.Linear(self.h_dim2, self.o_dim, bias=True)

    def forward(self, input):

        # Layer1
        x = input
        x = self.linear1(x)
        x = self.act1(x)
        x = self.ln1(x)

        # Layer2
        x = self.drop2(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.ln2(x)

        # Output
        x = self.drop_op(x)
        op = self.op(x)

        return op

class actor_nw(nn.Module):

    def __init__(self, ip_dim, num_action):
        super(actor_nw, self).__init__()
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

class diff_max(object):

    def __init__(self, dim_c=None, lg=None, dir_name=None, pro_folder=None):
        self.dim_c = dim_c
        self.lg = lg
        self.dir_name = dir_name
        self.pro_folder = pro_folder
        self.agent = AGENT
        self.action_space = [ _ for _ in range(NUM_ACTIONS)]

        # ------------------------- #
        np.random.seed(SEED)
        tc.manual_seed(SEED)
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
            self.actor = tc.load(self.pro_folder + '/load_model'+ '/model_'  + self.agent +"_"+str(ep) +".pt")
            self.actor.eval()
        else:
            # Reward Network
            ip_dim = (GRID * GRID)*(GRID * GRID) * NUM_ACTIONS
            self.rt_nw = rt_nw(ip_dim)
            self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=LEARNING_RATE)

            # Policy Network
            ip_dim = (GRID * GRID)*(GRID * GRID)
            self.actor = actor_nw(ip_dim, NUM_ACTIONS)
            self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()

        self.max_return = -HUGE

    def action(self, obs, nts):
        act_n = []
        act_id = []
        with tc.no_grad():
            x = [nts.ravel()]
            pi_logit, _ = self.actor(x)
            prob = F.softmax(pi_logit).data.numpy()[0]
            for i in range(NUM_AGENTS):
                u = np.zeros(5)  # 5-d because of no-move action
                a = int(np.random.choice(self.action_space, 1, p=prob)[0])
                u[a] = 1
                act = np.concatenate([u, np.zeros(self.dim_c)])
                act_n.append(act)
                act_id.append(a)
        return act_n, act_id, prob

    def clear_buffer(self):

        self.data_pt = BATCH_SIZE * HORIZON
        self.buff_nts = np.zeros((self.data_pt, GRID*GRID * GRID*GRID), dtype=int)
        self.buff_ntsa = np.zeros((self.data_pt, GRID*GRID, GRID*GRID, NUM_ACTIONS), dtype=int)
        self.buff_ntsa_diff = np.zeros((self.data_pt, GRID*GRID*GRID*GRID*NUM_ACTIONS), dtype=int)
        self.batch_id = 0
        self.buff_diff_return = np.zeros((self.data_pt))
        self.buff_rt = np.zeros(self.data_pt)

    def store_rollouts(self, buff_nts=None, buff_ntsa=None, buff_rt=None):

        st = self.batch_id * HORIZON
        en = st + HORIZON
        # set_trace()
        self.buff_nts[st:en] = buff_nts.reshape(HORIZON, GRID*GRID * GRID*GRID)
        self.buff_ntsa[st:en] = buff_ntsa.reshape(HORIZON, GRID*GRID , GRID*GRID, NUM_ACTIONS)

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt

        # -------- Input for rw
        nsa = buff_ntsa.reshape(HORIZON, GRID*GRID * GRID*GRID * NUM_ACTIONS)

        self.buff_ntsa_diff[st:en] = nsa
        # -------- Diff Reward
        input = tc.FloatTensor(nsa)
        input = input / NUM_AGENTS
        input.requires_grad = True
        rt_nn = self.rt_nw(input).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/NUM_AGENTS

        # Default sa grad
        # mid_point = int((GRID*GRID * GRID*GRID * NUM_ACTIONS)/2)
        max_point = -1
        def_grad = grad[:, max_point].reshape(HORIZON, 1)
        grad = grad - def_grad
        diff_rw = grad.numpy()
        diff_all = diff_rw * nsa
        d_rt = diff_all.sum(1)
        self.buff_diff_return[st:en] = discounted_return(d_rt)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # set_trace()
        # ------- Reward Network
        x_tmp = self.buff_ntsa_diff.copy()
        x_final = x_tmp/NUM_AGENTS
        r_final = self.buff_rt.copy()
        loss_list = []
        for i in range(EPOCH):
            x = tc.FloatTensor(x_final)
            y_pred = self.rt_nw(x).squeeze()
            y_target = tc.tensor(r_final).float()
            loss = F.mse_loss(y_pred, y_target)
            self.rt_nw_opt.zero_grad()
            loss.backward()
            self.rt_nw_opt.step()
            loss_tmp = loss.data.numpy()
            loss_list.append(loss_tmp)
        self.loss = np.mean(loss_list)
        self.writer.add_scalar('Loss' + '/total', self.loss, ep)

        # ------- Policy Network
        if ep > POLICY_TRAIN:
            input = self.buff_nts
            pi_logit, log_pi = self.actor(input)
            action_probs = F.softmax(pi_logit)
            dist = Categorical(action_probs)
            entropy = dist.entropy()
            ntsa = self.buff_ntsa
            ntsa = tc.tensor(ntsa.sum(1).sum(1)).float()
            op1 = tc.mul(ntsa, log_pi).sum(1)
            gt = tc.tensor(self.buff_diff_return).float()
            op2 = tc.mul(op1, gt)
            op4 = tc.add(op2, ENTROPY_WEIGHT * entropy)
            pi_loss = -(tc.mean(op4))
            self.actor_opt.zero_grad()
            pi_loss.backward()
            self.actor_opt.step()

    def log(self, ep, ep_rw, buff_act_prob, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        if LOCAL:
            # --------- Reward Nw Weights
            self.writer.add_histogram('Rw_Weight/' + "l1", self.rt_nw.linear1.weight, ep)
            self.writer.add_histogram('Rw_Weight/' + "l2", self.rt_nw.linear2.weight, ep)
            self.writer.add_histogram('Rw_Weight/' + "op", self.rt_nw.op.weight, ep)

            # -------- Policy Nw Weights
            self.writer.add_histogram('Policy_Weight/l1', self.actor.linear1.weight, ep)
            self.writer.add_histogram('Policy_Weight/l2', self.actor.linear2.weight, ep)
            self.writer.add_histogram('Policy_Weight/pi', self.actor.pi.weight, ep)
        # ---- Entropy
        pr = tc.tensor(buff_act_prob)
        entr = Categorical(probs=pr).entropy()
        entr_mean = entr.mean()
        self.writer.add_scalar('Entropy/', entr_mean, ep)
        # Action Count
        for a in range(NUM_ACTIONS):
            self.writer.add_scalar('Mean_Action_Count' + "/" + str(a), mean_act_count[a], ep)

    def save_model(self, tot_reward, ep):

        if tot_reward >= self.max_return:
            self.max_return = tot_reward
            cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
            os.system(cmd+"*.*")
            dumpDataStr(self.pro_folder+'/log/'+self.dir_name+'/model/'+self.agent+'_max_ep', ep)
            # Reward Network
            tc.save(self.rt_nw, self.pro_folder + '/log/' + self.dir_name + '/model/model_rt' + "_" + self.agent + "_"+str(ep) +".pt")
            # Policy Network
            tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor' + "_" + self.agent + "_" + str(ep) + ".pt")

        # Convergence
        # Reward Network
        tc.save(self.rt_nw, self.pro_folder + '/log/' + self.dir_name + '/model/model_rt' + "_" + self.agent+".pt")
        # Policy Network
        tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor' + "_" + self.agent + ".pt")

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    print("# ============================  END  ============================ #")