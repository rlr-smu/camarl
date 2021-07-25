"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 16 Jul 2018
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
import numpy as np
from data import totalZONES, HORIZON, listHORIZON, planningZONES, totalVESSELS, dummyZONES, termZONES, Mask, planningTermZones, dirName, T_min_max, minTmin, maxTmax

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
from tensorboardX import SummaryWriter
# =============================== Variables ================================== #

# ============================================================================ #
class tminAgent:

    def __init__(self, instance, load_model):

        self.instance = instance
        self.epoch = 0
        self.writer = SummaryWriter()
        return
    def getBeta(self, cT, i_episode):

        return np.zeros((1, totalZONES, totalZONES))

    def train(self, i_episode):

        return

    def storeRollouts(self, buffer_nt_z, buffer_ntz_zt, buffer_rt_z, buffer_beta, buffer_nt_ztz, buffer_rt):

        return

    def clear(self):
        return

    def log(self, i_episode, epReward, betaAvg2, sampleAvg):

        return
    def save_model(self, i_episode):

        return

    def ep_init(self):
        return

# =============================================================================== #

if __name__ == '__main__':
    main()
    