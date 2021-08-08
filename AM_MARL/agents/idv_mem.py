"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 03 Jun 2021
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
import pdb
import rlcompleter
import numpy as np
from agents.base_agent import baseAgent, baseAgent_dec
from agents.networks import actor_nw, actor_dec_nw
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, HORIZON, ENTROPY_WEIGHT, VERBOSE, SEED, LAMBDA, TINY, TOTAL_AGENTS
import torch as tc
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import discounted_return
import numba as nb
from utils import compute_idv

# =============================== Variables ================================== #

tc.random.manual_seed(SEED)
# ============================================================================ #

class idv_mem(baseAgent):

    def __init__(self, config=None):
        super(idv_mem, self).__init__(config=config)

        self.actor = None
        if LOAD_MODEL:
            self.lg.writeln("-----------------------")
            self.lg.writeln("Loading Old Model")
            self.lg.writeln("-----------------------")
            self.actor = tc.load(self.pro_folder + '/load_model'+ '/model_actor_'  + self.agent_name +".pt")
            self.actor.eval()
        else:
            # Policy Network
            # ip_dim = self.s_dim + self.o_dim
            # ip_dim = self.num_states + self.o_dim
            ip_dim = self.one_hot_dim + self.o_dim
            self.actor = actor_nw(ip_dim, self.num_actions)
            self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.entropy = -1
        self.epoch = 1
        self.meanAds = 0
        self.stdAds = 0


    def compute_val_fn(self, buff_rt=None, buff_at_rw=None, buff_ntsa=None, buff_ntsas=None):

        horizon = buff_at_rw.shape[0]
        q_tmp = np.zeros((horizon, self.num_states, self.num_actions))

        rewards = buff_at_rw
        counts = buff_ntsas
        action_counts = buff_ntsa

        state_r  = np.zeros(rewards.shape[1])
        transits = counts/np.maximum(action_counts, 1e-8)[:,:,:,np.newaxis]
        action_probs = action_counts/np.maximum(action_counts.sum(axis = 2),1e-8)[:,:,np.newaxis]

        for t in range(horizon - 1, -1, -1):
            reward = rewards[t]
            transit = transits[t]
            action_prob = action_probs[t]
            r = reward + DISCOUNT*np.sum(transit*((state_r)[np.newaxis, np.newaxis,:]), axis=2) # fixed off by one bug
            state_r = (r * action_prob).sum(axis=1)
            q_tmp[t] = r

        # if self.epoch == 1:
        #     self.meanAds = np.sum(q_tmp * buff_ntsa) / horizon
        #     self.stdAds = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buff_ntsa) / horizon)
        # else:
        #     self.meanAds1 = np.sum(q_tmp * buff_ntsa) / horizon
        #     self.stdAds1 = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buff_ntsa) / horizon)
        #
        #     self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
        #     self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds
        #
        # q_tmp = (q_tmp - self.meanAds) / (self.stdAds + TINY)
        # v_tmp = q_tmp * buff_ntsa
        # v_tmp[:,:,:] = np.sum(v_tmp, axis=2)[:,:,np.newaxis]
        # # ------------ Advantage ------------ #
        # Adv = q_tmp - v_tmp

        return q_tmp

    def train(self, x_mem=None, v_mem=None, n_mem=None, g_mem=None):

        # ----- Policy Train
        input = x_mem
        s = input[:, 0].astype(np.int32)
        s_1hot = self.one_hot_hash[s]
        input_1hot = np.hstack((s_1hot, input[:, 1:]))
        nsa = tc.tensor(n_mem).float()

        # --- normalize count
        nsa = nsa / TOTAL_AGENTS

        # ---- val_fn
        val = tc.tensor(v_mem + LAMBDA*g_mem).float()
        pi_logit, log_pi = self.actor(input_1hot)
        action_probs = F.softmax(pi_logit)
        dist = Categorical(action_probs)
        entropy = dist.entropy().reshape(input.shape[0], 1)
        op1 = tc.mul(nsa, log_pi)
        op2 = tc.mul(op1, val)
        op3 = tc.add(op2, ENTROPY_WEIGHT * entropy)
        pi_loss = -(tc.mean(op3))
        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()
        self.entropy = entropy.mean().data

        self.epoch += 1

    def log_agent(self, ep,  node_len, avg_nbr, graph_hit):

        # ---- Other stats
        self.writer.add_scalar('OtherStats/node_size', node_len, ep)
        self.writer.add_scalar('OtherStats/avg_nbr', avg_nbr, ep)
        self.writer.add_scalar('OtherStats/graph_hit', graph_hit, ep)

        if VERBOSE:
            self.writer.add_histogram('ac_weight/' + "l1", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "l2", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "pi", self.actor.pi.weight, ep)

class idv_dec_mem(baseAgent_dec):

    def __init__(self, config=None):
        super(idv_dec_mem, self).__init__(config=config)

        self.actor = None
        if LOAD_MODEL:
            self.lg.writeln("-----------------------")
            self.lg.writeln("Loading Old Model")
            self.lg.writeln("-----------------------")
            self.actor = tc.load(self.pro_folder + '/load_model'+ '/model_actor_'  + self.agent_name +".pt")
            self.actor.eval()
        else:
            # Policy Network
            ip_dim = self.one_hot_dim + self.o_dim
            self.actor = actor_dec_nw(ip_dim, self.num_states, self.num_actions, self.o_nbr_index, self.one_hot_hash, self.o_dim)
            self.actor_opt = tc.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.entropy = -1
        self.epoch = 1
        self.meanAds = 0
        self.stdAds = 0

    def compute_val_fn(self, buff_rt=None, buff_at_rw=None, buff_ntsa=None, buff_ntsas=None):

        horizon = buff_at_rw.shape[0]
        q_tmp = np.zeros((horizon, self.num_states, self.num_actions))

        rewards = buff_at_rw
        counts = buff_ntsas
        action_counts = buff_ntsa

        state_r  = np.zeros(rewards.shape[1])
        transits = counts/np.maximum(action_counts, 1e-8)[:,:,:,np.newaxis]
        action_probs = action_counts/np.maximum(action_counts.sum(axis = 2),1e-8)[:,:,np.newaxis]

        for t in range(horizon - 1, -1, -1):
            reward = rewards[t]
            transit = transits[t]
            action_prob = action_probs[t]
            r = reward + DISCOUNT*np.sum(transit*((state_r)[np.newaxis, np.newaxis,:]), axis=2) # fixed off by one bug
            state_r = (r * action_prob).sum(axis=1)
            q_tmp[t] = r

        # if self.epoch == 1:
        #     self.meanAds = np.sum(q_tmp * buff_ntsa) / horizon
        #     self.stdAds = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buff_ntsa) / horizon)
        # else:
        #     self.meanAds1 = np.sum(q_tmp * buff_ntsa) / horizon
        #     self.stdAds1 = np.sqrt(np.sum(np.square(q_tmp - self.meanAds) * buff_ntsa) / horizon)
        #
        #     self.meanAds = 0.9 * self.meanAds1 + 0.1 * self.meanAds
        #     self.stdAds = 0.9 * self.stdAds1 + 0.1 * self.stdAds
        #
        # q_tmp = (q_tmp - self.meanAds) / (self.stdAds + TINY)
        # v_tmp = q_tmp * buff_ntsa
        # v_tmp[:,:,:] = np.sum(v_tmp, axis=2)[:,:,np.newaxis]
        # # ------------ Advantage ------------ #
        # Adv = q_tmp - v_tmp

        return q_tmp

    def train(self, x_mem=None, v_mem=None, n_mem=None, g_mem=None):

        # ----- Policy Train
        input = x_mem
        nsa = tc.tensor(n_mem).float()

        # --- normalize count
        nsa = nsa / TOTAL_AGENTS

        # ---- val_fn
        val = tc.tensor(v_mem + LAMBDA*g_mem).float()

        pi_logit, log_pi = self.actor(input)
        action_probs = F.softmax(pi_logit, -1)
        dist = Categorical(action_probs)

        entropy = dist.entropy().reshape(nsa.shape[0], nsa.shape[1], 1)
        op1 = tc.mul(nsa, log_pi)
        op2 = tc.mul(op1, val)
        op3 = tc.add(op2, ENTROPY_WEIGHT * entropy)


        pi_loss = -(tc.mean(op3))


        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()
        self.entropy = entropy.mean().data

        self.epoch += 1

    # def log_agent(self, ep,  node_len, avg_nbr, fg_val):
    def log_agent(self, ep, node_len, avg_nbr):
        # ---- Other stats
        self.writer.add_scalar('OtherStats/node_size', node_len, ep)
        self.writer.add_scalar('OtherStats/avg_nbr', avg_nbr, ep)
        # self.writer.add_scalar('OtherStats/graph_hit', graph_hit, ep)

        # for i in range(fg_val.shape[0]):
        #     self.writer.add_scalar('fg_val/'+str(i), fg_val[i][0], ep)

        if VERBOSE:
            self.writer.add_histogram('ac_weight/' + "l1", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "l2", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "pi", self.actor.pi.weight, ep)

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
