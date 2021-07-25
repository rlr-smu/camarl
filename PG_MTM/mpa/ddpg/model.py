"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 29 Jun 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print "# ============================ START ============================ #"
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
import torch as tc
import numpy as np
from torch.distributions.normal import Normal
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
# =============================== Variables ================================== #

torch.manual_seed(SEED)
np.random.seed(SEED)
tc.set_num_threads(NUM_CORES)
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

        self.sigmoid2 = nn.ModuleList()

        self.Mask = torch.FloatTensor(Mask)
        for z in range(totalZONES):
            self.linear1.append(nn.Linear(totalZONES, totalZONES, bias=True))
            torch.nn.init.xavier_uniform(self.linear1[z].weight)
            self.tanh1.append(nn.Tanh())
            self.ln1.append(nn.LayerNorm(totalZONES))
            self.linear2.append(nn.Linear(totalZONES, totalZONES, bias=True))
            torch.nn.init.xavier_uniform(self.linear2[z].weight)
            self.tanh2.append(nn.Tanh())
            self.ln2.append(nn.LayerNorm(totalZONES))
            self.op.append(nn.Linear(totalZONES, totalZONES, bias=True))
            torch.nn.init.xavier_uniform(self.op[z].weight)
            self.sigmoid.append(nn.Sigmoid())
            self.sigmoid2.append(nn.Sigmoid())

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

    def forward(self, x, epsilon):


        dtPt = x.shape[0]
        x = torch.FloatTensor(x)
        tmpOp = torch.tensor([])
        self.normal = Normal(tc.tensor([0.0]), epsilon)

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
            # noise = self.normal.sample(x.shape).view(x.shape)
            # x = x + noise
            # x = self.sigmoid2[z](x)
            tmpOp = torch.cat((tmpOp, x), 1)

        if np.random.uniform() > epsilon:
            tmpOp = torch.reshape(tmpOp, (dtPt, totalZONES, totalZONES))
        else:
            tmpOp = tc.FloatTensor(dtPt, totalZONES, totalZONES).uniform_()

        return tmpOp


class critic(nn.Module):

    def __init__(self):
        super(critic, self).__init__()

        self.iDim = 4  # < nt(z), nt(z'), nt(z, z', phi), beta_zz >
        self.hDim_1 = 2 * self.iDim
        self.hDim_2 = 2 * self.iDim

        self.linear1 = nn.ModuleList()
        self.tanh1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.tanh2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.linear3 = nn.ModuleList()

        layerIdx = 0
        for z in range(totalZONES):
            for zp in range(totalZONES):
                self.linear1.append(nn.Linear(self.iDim, self.hDim_1, bias=True))
                torch.nn.init.xavier_uniform(self.linear1[layerIdx].weight)
                self.tanh1.append(nn.Tanh())
                self.ln1.append(nn.LayerNorm(self.hDim_1))
                self.linear2.append(nn.Linear(self.hDim_1 + 1, self.hDim_2, bias=True))
                torch.nn.init.xavier_uniform(self.linear2[layerIdx].weight)
                self.tanh2.append(nn.Tanh())
                self.ln2.append(nn.LayerNorm(self.hDim_2))
                self.linear3.append(nn.Linear(self.hDim_2 + 1, 1, bias=True))
                torch.nn.init.xavier_uniform(self.linear3[layerIdx].weight)
                layerIdx += 1

    def forward(self, nt, ntzz, beta):

        nt = tc.tensor(nt).float()
        dtPt = nt.shape[0]
        layerIdx = 0
        output = torch.tensor([])
        for z in range(totalZONES):
            local_output = []
            for zp in range(totalZONES):
                x = []
                x.append(nt[:, z])
                x.append(nt[:, zp])
                x.append(ntzz[:, z, zp])
                x.append(beta[:, z, zp])
                x = tc.stack(x, 1)
                x = self.linear1[layerIdx](x)
                x = self.tanh1[layerIdx](x)
                x = self.ln1[layerIdx](x)
                t1 = beta[:, z, zp].view(-1, 1)
                x = tc.cat((x, t1), 1)
                x = self.linear2[layerIdx](x)
                x = self.tanh2[layerIdx](x)
                x = self.ln2[layerIdx](x)
                t1 = beta[:, z, zp].view(-1, 1)
                x = tc.cat((x, t1), 1)
                x = self.linear3[layerIdx](x)
                moving = tc.reshape(ntzz[:, z, zp], (dtPt, 1))
                x = x * moving
                local_output.append(x)
                layerIdx += 1
            local_output = tc.stack(local_output, 1)
            output = tc.cat((output, local_output), 1)
        output = tc.reshape(output, (dtPt, totalZONES, totalZONES))
        return output

# =============================================================================== #
