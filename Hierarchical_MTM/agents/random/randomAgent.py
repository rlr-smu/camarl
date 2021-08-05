"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 14 Apr 2019
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
import auxLib as ax
import pdb
import rlcompleter
import numpy as np
from tensorboardX import SummaryWriter
import networkx as nx
from numpy.random import multinomial
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

class randomAgent:

    def __init__(self, load_model=None, dirName=None, data=None):

        self.totalZONES = data.totalZONES
        self.dummyZONES = data.dummyZONES
        self.termZONES = data.termZONES
        self.planningZONES = data.planningZONES
        self.zGraph = data.zGraph
        self.num_actions = data.num_actions
        self.action_space = data.action_space
        self.epoch = -1
        self.writer = SummaryWriter()
        self.p = np.zeros((self.totalZONES, self.totalZONES, self.num_actions))
        for z in range(self.totalZONES):
            for zp in range(self.totalZONES):
                self.p[z][zp] = [1.0 / self.num_actions for _ in range(self.num_actions)]


    def getAction(self, t, cT, i_episode):

        # ----- Get Action Probability from Policy Network
        action_prob = self.p
        return action_prob

    def train(self, i_episode):

        return

    def storeRollouts(self, buffer_nt_z,  buffer_rt_z, buffer_beta, buffer_rt):

        return

    def clear(self):
        return
    
    def log(self, i_episode, epReward, betaAvg2, vio, delay):

            self.writer.add_scalar('Total Rewards', epReward, i_episode)
            self.writer.add_scalar('Total ResVio', vio, i_episode)
            self.writer.add_scalar('Total Delay', delay, i_episode)

            # self.writer.add_scalar('Sample Avg. Rewards', sampleAvg, i_episode)
            # self.writer.add_scalar('Loss', self.loss, i_episode)
            for z in range(self.totalZONES):
                if (z not in self.dummyZONES) and (z not in self.termZONES):
                    for zp in nx.neighbors(self.zGraph, z):
                        self.writer.add_scalar("Beta/"+str(z)+"_"+str(zp), betaAvg2[z][zp], i_episode)
    
    def save_model(self):

        return

    def ep_init(self):
        return




# =============================================================================== #

if __name__ == '__main__':
    main()
    