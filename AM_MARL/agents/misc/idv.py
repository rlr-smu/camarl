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
from agents.base_agent import baseAgent
from agents.networks import actor_nw
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

class idv(baseAgent):

    def __init__(self, config=None):
        super(idv, self).__init__(config=config)

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

        val_fn, meanAds, stdAds = compute_idv(buff_at_rw, buff_ntsa, self.epoch, self.num_states, self.num_actions, self.meanAds, self.stdAds)
        self.meanAds = meanAds
        self.stdAds = stdAds
        return val_fn

    def train(self, x_mem=None, v_mem=None, n_mem=None, g_mem=None):

        # ----- Policy Train
        input = x_mem
        s = input[:, 0].astype(np.int32)
        s_1hot = self.one_hot_hash[s]
        input_1hot = np.hstack((s_1hot, input[:, 1:]))
        nsa = tc.tensor(n_mem).float()

        # --- normalize count
        nsa = nsa / TOTAL_AGENTS

        # ---- val fn
        val = tc.tensor(v_mem).float()
        pi_logit, log_pi = self.actor(input_1hot)
        action_probs = F.softmax(pi_logit)
        dist = Categorical(action_probs)
        entropy = dist.entropy().reshape(input.shape[0], 1)
        op1 = tc.mul(nsa, log_pi)
        op2 = tc.mul(op1, val)
        set_trace()
        op3 = tc.add(op2, ENTROPY_WEIGHT * entropy)
        pi_loss = -(tc.mean(op3))
        self.actor_opt.zero_grad()
        pi_loss.backward()
        self.actor_opt.step()
        self.entropy = entropy.mean().data

        self.epoch += 1

    def log_agent(self, ep):

        if VERBOSE:
            self.writer.add_histogram('ac_weight/' + "l1", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "l2", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "pi", self.actor.pi.weight, ep)

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
