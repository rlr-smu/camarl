"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 27 Jun 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import auxLib as ax
import pdb
import rlcompleter
from data import totalZONES, HORIZON, listHORIZON, planningZONES, totalVESSELS, dummyZONES, termZONES, Mask, planningTermZones, dirName, T_min_max, minTmin, maxTmax
from parameters import LOAD_MODEL, TINY, SAVE_MODEL, SEED, KEEP_MODELS, MAX_BINARY_LENGTH, LEARNING_RATE, OPTIMIZER, BATCH_SIZE, DISCOUNT, SHOULD_LOG, MAP_ID, VF_NORM, NUM_CORES
import torch
import torch.nn as nn
# import torch.nn.functional

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

ipDim = totalZONES #(totalVESSELS + 1) * totalZONES
h1Dim = totalZONES
h2Dim = totalZONES
opDim = totalZONES
torch.manual_seed(SEED)
torch.set_num_threads(NUM_CORES)


# ============================================================================ #

class actor(nn.Module):

    def __init__(self):
        super(actor, self).__init__()
        self.linear1 = nn.ModuleList()
        self.tanh1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.tanh2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.op = nn.ModuleList()
        self.sigmoid = nn.ModuleList()

        self.Mask = torch.FloatTensor(Mask)
        # self.myparameters = nn.ParameterList(self.parameters())

        for z in range(totalZONES):
            self.linear1.append(nn.Linear(ipDim, h1Dim, bias=True))
            torch.nn.init.xavier_uniform(self.linear1[z].weight)
            # torch.nn.init.xavier_uniform(self.linear1[z].bias)
            self.tanh1.append(nn.Tanh())
            self.ln1.append(nn.LayerNorm(h1Dim))
            self.linear2.append(nn.Linear(h1Dim, h2Dim, bias=True))
            torch.nn.init.xavier_uniform(self.linear2[z].weight)
            # torch.nn.init.xavier_uniform(self.linear2[z].bias)
            self.tanh2.append(nn.Tanh())
            self.ln2.append(nn.LayerNorm(h2Dim))
            self.op.append(nn.Linear(h2Dim, opDim, bias=True))
            torch.nn.init.xavier_uniform(self.op[z].weight)
            self.sigmoid.append(nn.Sigmoid())
            # torch.nn.init.xavier_uniform(self.op[z].bias)

    def forward(self, x, dtPt):

        x = torch.FloatTensor(x)
        tmpOp = torch.tensor([])
        for z in range(totalZONES):
            x = torch.mul(x, self.Mask[z])
            x = self.linear1[z](x)
            x = self.tanh1[z](x)
            x = self.ln1[z](x)
            x = self.linear2[z](x)
            x = self.tanh2[z](x)
            x = self.ln2[z](x)
            x = self.op[z](x)
            x = self.sigmoid[z](x)
            tmpOp = torch.cat((tmpOp, x), 1)
        tmpOp = torch.reshape(tmpOp, (dtPt, totalZONES, totalZONES))
        return tmpOp

# =============================================================================== #
