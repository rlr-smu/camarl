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
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, NUM_CORES, POLICY_TRAIN, EPOCH, DROPOUT, SHUFFLE, SEED, AGENT, GRID, NUM_AGENTS, NUM_ACTIONS, HORIZON, BATCH_SIZE, ENTROPY_WEIGHT, LOCAL, HUGE, CSA
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

class appx_dr_colby(object):

    def __init__(self, dim_c=None, lg=None, dir_name=None, pro_folder=None):
        self.dim_c = dim_c
        self.lg = lg
        self.dir_name = dir_name
        self.pro_folder = pro_folder
        self.agent = AGENT
        self.action_space = [ _ for _ in range(NUM_ACTIONS)]
        self.vel = 2
        self.pos = 2
        self.entity_pos = 2 * (NUM_AGENTS)
        self.other_pos = 2 * (NUM_AGENTS - 1)
        self.coms = 2 * (NUM_AGENTS - 1)
        self.num_actions = NUM_ACTIONS


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

            self.rw_nw_list = []
            self.rw_opt_list = []
            self.actor_list = []
            self.act_opt_list = []

            for i in range(NUM_AGENTS):
                # Actor Network
                ip_dim = (GRID * GRID)*(GRID * GRID)


                self.actor_list.append(actor_nw(ip_dim, self.num_actions))
                self.act_opt_list.append(tc.optim.Adam(self.actor_list[i].parameters(), lr=LEARNING_RATE))

                # Reward Network
                ip_dim = (GRID * GRID) * (GRID * GRID) + 1
                self.rw_nw_list.append(rt_nw(ip_dim))
                self.rw_opt_list.append(tc.optim.Adam(self.rw_nw_list[i].parameters(), lr=LEARNING_RATE))


        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()

        self.max_return = -HUGE

        # --- DR
        if CSA == "mid":
            self.csa_id = int((GRID * GRID) * (GRID*GRID)/2)
        elif CSA == "max":
            self.csa_id = int((GRID * GRID) * (GRID * GRID) - 1)
        self.state_dim = int((GRID * GRID) * (GRID*GRID))

    def action(self, obs, nts):

        act_n = []
        act_id = []
        with tc.no_grad():
            x = [nts.ravel()]
            for i in range(NUM_AGENTS):
                pi_logit, _ = self.actor_list[i](x)
                prob = F.softmax(pi_logit).data.numpy().squeeze()
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
        # self.buff_diff_return = np.zeros((self.data_pt))
        self.buff_rt = np.zeros(self.data_pt)

        self.local_buffer = {}
        self.buff_diff_return = {}
        for i in range(NUM_AGENTS):
            self.local_buffer[i] = []
            self.buff_diff_return[i] = []

    def update_buffer(self, t, nts, act, rt):

        for i in range(NUM_AGENTS):
            s = nts.ravel()
            a = act[i]
            x = np.hstack((t, s, a, rt))
            self.local_buffer[i].append(x)

    def store_rollouts(self, buff_nts=None, buff_ntsa=None, buff_rt=None):

        st = self.batch_id * HORIZON
        en = st + HORIZON
        ds_default = np.zeros((HORIZON, (GRID*GRID)*GRID*GRID))

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt

        for i in self.local_buffer:
            x_all = np.array(self.local_buffer[i])[st:en]
            x = x_all[:,1:self.state_dim+1]
            a_list = x_all[:,-2:-1].astype('int')
            ds_count = x[:, self.csa_id]
            ds_default[:, self.csa_id] = ds_count
            x_r = tc.tensor(np.hstack((ds_default, a_list))).float()
            countFac = self.rw_nw_list[i](x_r).data.numpy().squeeze()
            dr = buff_rt - countFac
            dr_return = discounted_return(dr)
            self.buff_diff_return[i].extend(dr_return)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # loss = nn.MSELoss(reduction='mean')
        loss_list = []

        for i in self.local_buffer:
            x_all = np.array(self.local_buffer[i])

            # ---- Reward Nw
            sa_rw = tc.tensor(x_all[:, 1:self.state_dim + 2]).float()
            y_pred = self.rw_nw_list[i](sa_rw).squeeze(-1)
            y_target = tc.tensor(self.buff_rt).float()
            # loss_i = loss(y_pred, y_target)
            op1 = y_pred - y_target
            op2 = op1 * op1
            dtpt = op1.shape[0]
            op3 = op2.sum()
            loss_i = op3/dtpt

            self.rw_opt_list[i].zero_grad()
            loss_i.backward()
            self.rw_opt_list[i].step()
            loss_list.append(loss_i.data.numpy())

            # ----- Policy Network
            if ep > POLICY_TRAIN:
                s_pi = tc.tensor(x_all[:, 1:self.state_dim + 1]).float()
                _, log_pi = self.actor_list[i](s_pi)
                gt = tc.tensor(self.buff_diff_return[i]).float().unsqueeze(-1)
                op1 = tc.mul(log_pi, gt).sum(-1)
                pi_loss = tc.mean(op1)
                self.act_opt_list[i].zero_grad()
                pi_loss.backward()
                self.act_opt_list[i].step()

        self.loss = np.mean(loss_list)
        self.writer.add_scalar('Loss' + '/total', self.loss, ep)



    def log(self, ep, ep_rw, buff_act_prob, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)


        # if LOCAL:
        #     # --------- Reward Nw Weights
        #     self.writer.add_histogram('Rw_Weight/' + "l1", self.rt_nw.linear1.weight, ep)
        #     self.writer.add_histogram('Rw_Weight/' + "l2", self.rt_nw.linear2.weight, ep)
        #     self.writer.add_histogram('Rw_Weight/' + "op", self.rt_nw.op.weight, ep)
        #
        #     # -------- Policy Nw Weights
        #     self.writer.add_histogram('Policy_Weight/l1', self.actor.linear1.weight, ep)
        #     self.writer.add_histogram('Policy_Weight/l2', self.actor.linear2.weight, ep)
        #     self.writer.add_histogram('Policy_Weight/pi', self.actor.pi.weight, ep)
        # # ---- Entropy
        # pr = tc.tensor(buff_act_prob)
        # entr = Categorical(probs=pr).entropy()
        # entr_mean = entr.mean()
        # self.writer.add_scalar('Entropy/', entr_mean, ep)
        # # Action Count
        # for a in range(NUM_ACTIONS):
        #     self.writer.add_scalar('Mean_Action_Count' + "/" + str(a), mean_act_count[a], ep)

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