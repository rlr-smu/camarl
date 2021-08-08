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
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from parameters import HIDDEN_DIM, SEED


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #

tc.manual_seed(SEED)


# ============================================================================ #

class actor_dec_nw(nn.Module):

    def __init__(self, ip_dim, num_states, num_actions, o_nbr_index, one_hot_hash, o_dim):
        super(actor_dec_nw, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.o_nbr_index = o_nbr_index
        self.one_hot_hash = one_hot_hash
        self.o_dim = o_dim
        self.iDim = ip_dim
        self.hDim_1 = HIDDEN_DIM

        # L1
        self.linear1 = nn.ModuleList()
        self.act1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()

        # L2
        self.linear2 = nn.ModuleList()
        self.act2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()

        # Policy
        self.pi = nn.ModuleList()
        self.softmax = nn.ModuleList()
        self.lg_softmax = nn.ModuleList()

        for _ in range(self.num_states):
            # L1
            self.linear1.append(nn.Linear(self.iDim, self.hDim_1, bias=True))
            self.act1.append(nn.LeakyReLU())
            self.ln1.append(nn.LayerNorm(self.hDim_1))

            # L2
            self.linear2.append(nn.Linear(self.hDim_1, self.hDim_1, bias=True))
            self.act2.append(nn.LeakyReLU())
            self.ln2.append(nn.LayerNorm(self.hDim_1))

            # Policy
            self.pi.append(nn.Linear(self.hDim_1, self.num_actions, bias=True))
            self.softmax.append(nn.Softmax(dim=-1))
            self.lg_softmax.append(nn.LogSoftmax(dim=-1))

    def forward(self, x):

        nt = tc.tensor(x).float()
        dtpt = nt.shape[0]
        output_pi_logit = []
        output_lg_sm = []

        for s in range(self.num_states):

            # ----- Input Layer
            s_1hot = tc.tensor(self.one_hot_hash[s]).float()
            s_1hot = s_1hot.repeat(dtpt, 1)
            o = tc.zeros(dtpt, self.o_dim)
            o_indx = self.o_nbr_index[s]
            o_len = o_indx.shape[0]
            o1 = nt[:, o_indx]
            o[:, 0:o_len] = o1
            x = tc.cat((s_1hot, o), 1)

            x = self.linear1[s](x)
            x = self.act1[s](x)
            x = self.ln1[s](x)

            # 2nd Layer
            x = self.linear2[s](x)
            x = self.act2[s](x)
            x = self.ln2[s](x)

            # Policy
            pi_logit = self.pi[s](x)
            lg_sm = self.lg_softmax[s](pi_logit)

            output_pi_logit.append(pi_logit)
            output_lg_sm.append(lg_sm)

        output_pi_logit = tc.stack(output_pi_logit, 1)
        output_lg_sm = tc.stack(output_lg_sm, 1)

        return output_pi_logit, output_lg_sm

class actor_nw(nn.Module):

    def __init__(self, ip_dim, num_action):
        super(actor_nw, self).__init__()
        self.iDim = ip_dim
        self.hDim_1 = HIDDEN_DIM

        # L1
        self.linear1 = nn.Linear(self.iDim, self.hDim_1, bias=True)
        self.act1 = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(self.hDim_1)

        # L2
        self.linear2 = nn.Linear(self.hDim_1, self.hDim_1, bias=True)
        self.act2 = nn.LeakyReLU()
        self.ln2 = nn.LayerNorm(self.hDim_1)

        # Policy
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

        # Policy
        pi_logit = self.pi(x)
        lg_sm = self.lg_softmax(pi_logit)
        return pi_logit, lg_sm

class critic_nw(nn.Module):

    def __init__(self, ip_dim, op_dim):
        super(critic_nw, self).__init__()
        self.iDim = ip_dim
        self.hDim_1 = HIDDEN_DIM

        # L1
        self.linear1 = nn.Linear(self.iDim, self.hDim_1, bias=True)
        self.act1 = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(self.hDim_1)

        # L2
        self.linear2 = nn.Linear(self.hDim_1, self.hDim_1, bias=True)
        self.act2 = nn.LeakyReLU()
        self.ln2 = nn.LayerNorm(self.hDim_1)

        # Q-Value
        self.q = nn.Linear(self.hDim_1, op_dim, bias=True)

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

        # Q-Value
        q = self.q(x)
        return q

def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
