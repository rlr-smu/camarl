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
from parameters import LOAD_MODEL, LEARNING_RATE, DISCOUNT, HORIZON, ENTROPY_WEIGHT, VERBOSE, SEED, LAMBDA, TOTAL_AGENTS
import torch as tc
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import discounted_return
import numba as nb
# =============================== Variables ================================== #

tc.random.manual_seed(SEED)
# ============================================================================ #

class vpg_mem(baseAgent):

    def __init__(self, config=None):
        super(vpg_mem, self).__init__(config=config)

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

    def compute_val_fn(self, buff_rt=None, buff_at_rw=None, buff_ntsa=None, buff_ntsas=None):

        # ------- Emperical Return ------- #
        tot_time = buff_rt.shape[0]
        empRet = discounted_return(buff_rt)
        empRet = empRet.reshape(tot_time, 1)

        # --- Compute value
        val_fn = np.zeros((tot_time, self.num_states * self.num_actions), dtype=np.float64)
        val_fn[:, ] = empRet
        val_fn = val_fn.reshape(tot_time, self.num_states, self.num_actions)
        return  val_fn

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

    def log_agent(self, ep,  node_len, avg_nbr, graph_hit):

        # ---- Other stats
        self.writer.add_scalar('OtherStats/node_size', node_len, ep)
        self.writer.add_scalar('OtherStats/avg_nbr', avg_nbr, ep)
        self.writer.add_scalar('OtherStats/graph_hit', graph_hit, ep)

        if VERBOSE:
            self.writer.add_histogram('ac_weight/' + "l1", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "l2", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "pi", self.actor.pi.weight, ep)

class vpg_dec_mem(baseAgent_dec):

    def __init__(self, config=None):
        super(vpg_dec_mem, self).__init__(config=config)

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

    def compute_val_fn(self, buff_rt=None, buff_at_rw=None, buff_ntsa=None, buff_ntsas=None):

        # ------- Emperical Return ------- #
        tot_time = buff_rt.shape[0]
        empRet = discounted_return(buff_rt)
        empRet = empRet.reshape(tot_time, 1)

        # --- Compute value
        val_fn = np.zeros((tot_time, self.num_states * self.num_actions), dtype=np.float64)
        val_fn[:, ] = empRet
        val_fn = val_fn.reshape(tot_time, self.num_states, self.num_actions)
        return  val_fn

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

    def log_agent(self, ep,  node_len, avg_nbr, graph_hit):

        # ---- Other stats
        self.writer.add_scalar('OtherStats/node_size', node_len, ep)
        self.writer.add_scalar('OtherStats/avg_nbr', avg_nbr, ep)
        self.writer.add_scalar('OtherStats/graph_hit', graph_hit, ep)

        if VERBOSE:
            self.writer.add_histogram('ac_weight/' + "l1", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "l2", self.actor.linear1.weight, ep)
            self.writer.add_histogram('ac_weight/' + "pi", self.actor.pi.weight, ep)

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
