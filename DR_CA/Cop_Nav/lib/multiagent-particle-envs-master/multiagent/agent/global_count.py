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

class q_nw(nn.Module):

    def __init__(self, ip_dim):
        super(q_nw, self).__init__()
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

class global_count(object):

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
            # Critic Network
            # --------------
            self.critic = tc.load(self.pro_folder + '/load_model' + '/model_critic' + "_" + self.agent +"_"+str(ep) +".pt")
            self.critic.eval()

            # Policy Network
            # --------------
            self.actor = tc.load(self.pro_folder + '/load_model'+ '/model_actor' + "_"  + self.agent +"_"+str(ep) +".pt")
            self.actor.eval()

        else:
            # Critic Network
            ip_dim = (GRID * GRID)*(GRID * GRID) * NUM_ACTIONS
            self.critic = q_nw(ip_dim)
            self.critic_opt = tc.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

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
        self.buff_act_prob = np.vstack((self.buff_act_prob, prob))
        return act_n, act_id, prob

    def clear_buffer(self):

        self.data_pt = BATCH_SIZE * HORIZON
        self.buff_nts = np.zeros((self.data_pt, GRID*GRID * GRID*GRID), dtype=int)
        self.buff_ntsa = np.zeros((self.data_pt, (GRID*GRID*GRID*GRID), NUM_ACTIONS), dtype=int)

        self.batch_id = 0
        self.buff_return = np.zeros((self.data_pt))
        self.buff_rt = np.zeros(self.data_pt)
        self.buff_act_prob = np.empty((0, NUM_ACTIONS))

    def store_rollouts(self, buff_nts=None, buff_ntsa=None, buff_rt=None):

        st = self.batch_id * HORIZON
        en = st + HORIZON
        self.buff_nts[st:en] = buff_nts.reshape(HORIZON, GRID*GRID * GRID*GRID)
        self.buff_ntsa[st:en] = buff_ntsa.reshape(HORIZON, GRID*GRID* GRID*GRID, NUM_ACTIONS)

        # -------- buff_rt
        self.buff_rt[st:en] = buff_rt

        # -------- Input for rw
        self.buff_return[st:en] = discounted_return(buff_rt)

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # ------- Critic Network
        dtpt = self.buff_ntsa.shape[0]

        x = self.buff_ntsa.reshape(dtpt, GRID*GRID* GRID*GRID * NUM_ACTIONS)
        x = tc.tensor(x).float()
        y_pred = self.critic(x).squeeze(1)
        y_target = tc.tensor(self.buff_return).float()
        loss = F.mse_loss(y_pred, y_target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self.writer.add_scalar('Loss' + '/total', loss.data.numpy(), ep)
        self.loss = loss.data.numpy()

        # ------- Policy Network
        if ep > POLICY_TRAIN:
            x = tc.tensor(self.buff_ntsa).float()
            x.requires_grad = True
            x_sum = x.sum(-1)
            pi = tc.tensor(self.buff_act_prob)
            op1 = x_sum.reshape(dtpt, GRID*GRID* GRID*GRID, 1)
            op2 = pi.reshape(dtpt, 1, NUM_ACTIONS)
            x_mean = (op1 * op2).float()
            x_mean = x_mean.reshape(dtpt, GRID*GRID* GRID*GRID * NUM_ACTIONS)
            x_mean = tc.FloatTensor(x_mean)
            qv = self.critic(x_mean).sum()
            q_grad = tc.autograd.grad(qv, x)[0].detach().numpy()
            q_grad = tc.tensor(q_grad)

            # ---- Fwd Pass
            input = tc.tensor(self.buff_nts)
            pi_logit, _ = self.actor(input)
            act_prob = F.softmax(pi_logit)
            act_prob = act_prob.reshape(dtpt, 1, NUM_ACTIONS)
            ntsa = tc.tensor(self.buff_ntsa)
            nts = ntsa.sum(-1)
            nts = nts.reshape(dtpt, GRID*GRID* GRID*GRID, 1)
            op1 = nts * act_prob
            op1 = op1.reshape(dtpt, GRID*GRID* GRID*GRID, NUM_ACTIONS)
            op2 = op1 * q_grad
            op3 = op2.sum(-1).sum(-1)
            pi_loss = -(tc.mean(op3))
            self.actor_opt.zero_grad()
            pi_loss.backward()
            self.actor_opt.step()

    def log(self, ep, ep_rw, buff_act_prob, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        if LOCAL:
            # --------- Critic Nw Weights
            self.writer.add_histogram('Q_Weight/' + "l1", self.critic.linear1.weight, ep)
            self.writer.add_histogram('Q_Weight/' + "l2", self.critic.linear2.weight, ep)
            self.writer.add_histogram('Q_Weight/' + "op", self.critic.op.weight, ep)

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
            # Critic Network
            tc.save(self.critic, self.pro_folder + '/log/' + self.dir_name + '/model/model_critic' + "_" + self.agent + "_"+str(ep) +".pt")
            # Policy Network
            tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor' + "_" + self.agent + "_" + str(ep) + ".pt")

        # Convergence
        # Critic Network
        tc.save(self.critic, self.pro_folder + '/log/' + self.dir_name + '/model/model_critic' + "_" + self.agent+".pt")
        # Policy Network
        tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor' + "_" + self.agent + ".pt")

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    print("# ============================  END  ============================ #")