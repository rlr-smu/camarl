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
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from ipdb import set_trace
from utils import discounted_return
from auxLib3 import dumpDataStr, loadDataStr
import numpy as np
# ============================================================================ #

class q_nw(nn.Module):

    def __init__(self, ip_dim, op_dim):
        super(q_nw, self).__init__()
        self.ip_dim = ip_dim
        self.h_dim1 = 100
        self.h_dim2 = 100
        self.o_dim = op_dim

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

class mean_field(object):

    def __init__(self, dim_c=None, lg=None, dir_name=None, pro_folder=None):

        self.dim_c = dim_c
        self.lg = lg
        self.dir_name = dir_name
        self.pro_folder = pro_folder
        self.agent = AGENT
        self.action_space = [ _ for _ in range(NUM_ACTIONS)]
        self.target_update = 10
        self.vel = 2
        self.pos = 2
        self.entity_pos = 2 * (NUM_AGENTS)
        self.other_pos = 2 * (NUM_AGENTS - 1)
        self.coms = 2 * (NUM_AGENTS - 1)

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
            ip_dim = NUM_AGENTS * (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms) + NUM_ACTIONS
            op_dim = NUM_ACTIONS
            self.critic = q_nw(ip_dim, op_dim)
            self.critic_opt = tc.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

            # Target Network
            self.critic_tgt = q_nw(ip_dim, op_dim)
            self.critic_tgt.load_state_dict(self.critic.state_dict())
            self.critic_tgt.eval()

            # Policy Network
            ip_dim = self.vel + self.pos + self.entity_pos + self.other_pos + self.coms
            self.actor = actor_nw(ip_dim, NUM_ACTIONS)
            self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        # -------- Create train parameters
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.clear_buffer()
        self.max_return = -HUGE

    def action(self, obs, _ ):

        act_n = []
        act_id = []
        x_c = np.empty((0, NUM_AGENTS * (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms) + NUM_ACTIONS))
        x = np.concatenate(obs)
        # tmp_x_c = []
        tmp_act = np.empty((0, NUM_ACTIONS))
        with tc.no_grad():
            for i in range(NUM_AGENTS):
                x_i = obs[i]
                self.x_a = np.vstack((self.x_a, x_i))
                pi_logit, _ = self.actor(x_i)
                prob = F.softmax(pi_logit).data.numpy()
                u = np.zeros(5)  # 5-d because of no-move action
                a = int(np.random.choice(self.action_space, 1, p=prob)[0])
                u[a] = 1

                self.a_c = np.vstack((self.a_c, a))

                act = np.concatenate([u, np.zeros(self.dim_c)])
                tmp_act = np.vstack((tmp_act, u))
                # tmp_x_c.append(np.concatenate((x, u)))
                act_n.append(act)
                act_id.append(a)
        mean_action = tmp_act.sum(0) / NUM_AGENTS
        for i in range(NUM_AGENTS):
            # t1 = tmp_x_c[i]
            t2 = np.hstack((x, mean_action))
            x_c = np.vstack((x_c, t2))
        self.x_c = np.vstack((self.x_c, x_c))

        return act_n, act_id, prob

    def get_action_p(self, obs):

        xp_c = np.empty((0, NUM_AGENTS * (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms) + NUM_ACTIONS))
        xp = np.concatenate(obs)
        # tmp_xp_c = []
        tmp_act = np.empty((0, NUM_ACTIONS))
        with tc.no_grad():
            for i in range(NUM_AGENTS):
                x_i = obs[i]
                self.xp_a = np.vstack((self.xp_a, x_i))
                pi_logit, _ = self.actor(x_i)
                prob = F.softmax(pi_logit).data.numpy()
                u = np.zeros(5)  # 5-d because of no-move action
                a = int(np.random.choice(self.action_space, 1, p=prob)[0])
                u[a] = 1
                tmp_act = np.vstack((tmp_act, u))
                # tmp_xp_c.append(np.concatenate((xp, u)))
        mean_action = tmp_act.sum(0) / NUM_AGENTS
        for i in range(NUM_AGENTS):
            # t1 = tmp_xp_c[i]
            t2 = np.hstack((xp, mean_action))
            xp_c = np.vstack((xp_c, t2))
        self.xp_c = np.vstack((self.xp_c, xp_c))

    def clear_buffer(self):

        self.data_pt = BATCH_SIZE * HORIZON
        self.buff_nts = np.zeros((self.data_pt, GRID*GRID * GRID*GRID), dtype=int)
        self.buff_ntsa = np.zeros((self.data_pt, GRID*GRID, GRID*GRID, NUM_ACTIONS), dtype=int)
        self.buff_ntsa_diff = np.zeros((self.data_pt, GRID*GRID*GRID*GRID*NUM_ACTIONS), dtype=int)
        self.batch_id = 0
        self.buff_diff_return = np.zeros((self.data_pt))
        self.buff_rt = []

        self.x_c = np.empty((0, NUM_AGENTS * (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms) + NUM_ACTIONS))
        self.a_c = np.empty((0, 1))
        self.x_a = np.empty((0, (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms)))

        self.xp_c = np.empty((0, NUM_AGENTS * (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms) + NUM_ACTIONS))
        self.xp_a = np.empty((0, (self.vel + self.pos + self.entity_pos + self.other_pos + self.coms)))

        self.buff_ac_act = np.empty((0, 1))

    def store_rollouts(self, buff_nts=None, buff_ntsa=None, buff_rt=None):

        st = self.batch_id * HORIZON
        en = st + HORIZON

        # -------- buff_rt
        for t in range(HORIZON):
            rt = buff_rt[t]
            t1 = [rt] * NUM_AGENTS
            self.buff_rt.extend(t1)

        # -------- Input for rw

        # -------- update batch id
        self.batch_id += 1

    def train(self, ep=None):

        # ------- Critic Network
        x_c = tc.tensor(self.x_c).float()
        y_pred_all = self.critic(x_c)
        act = tc.LongTensor(self.a_c)
        y_pred = y_pred_all.gather(1, act).squeeze(1)

        # Target
        xp_c = tc.tensor(self.xp_c).float()
        qp = self.critic_tgt(xp_c).detach()

        xp_a = tc.tensor(self.xp_a).float()
        pi_logit, _ = self.actor(xp_a)
        pp = F.softmax(pi_logit).data.numpy()

        v_mtmf = (qp * pp).sum(1).data.numpy()
        y_target = self.buff_rt + DISCOUNT * v_mtmf
        y_target = tc.tensor(y_target).float()
        loss = F.mse_loss(y_pred, y_target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self.writer.add_scalar('Loss' + '/total', loss.data.numpy(), ep)
        self.loss = loss.data.numpy()

        # ------- Policy Network
        if ep > POLICY_TRAIN:
            _, log_pi = self.actor(self.x_a)
            q_sa = y_pred_all.data.numpy()
            q_sa = tc.tensor(q_sa).float()
            # --- y_target
            op1 = log_pi * q_sa
            pi_loss = -tc.mean(op1)
            self.actor_opt.zero_grad()
            pi_loss.backward()
            self.actor_opt.step()

        # ----- Update Target Network
        if ep % self.target_update == 0:
            self.critic_tgt.load_state_dict(self.critic.state_dict())

    def log(self, ep, ep_rw, buff_act_prob, mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total_Rewards', ep_rw, ep)
        # self.writer.add_scalar('Reward/Total_Rewards_Agent', rt_sum_agent, ep)

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