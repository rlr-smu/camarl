"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 04 Mar 2020
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
from data import syn_data, real_data, real_real_data, real_real_data_eval
from parameters import NUM_CORES
import torch as tc
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# =============================== Variables ================================== #


# ============================================================================ #

tc.set_num_threads(NUM_CORES)

class data_agent(real_real_data_eval):

    def __init__(self, dir_name=None, pro_folder=None, lg=None):
        super(data_agent,  self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------------------------- #
        np.random.seed(self.seed)
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")
        self.action_prob = np.zeros((self.num_edges, self.num_actions))
        self.clear_buffer()

    def get_action(self, state=None):

        return self.max_action()

    def max_action(self):

        t1 = np.zeros((self.num_edges, self.num_actions))
        t1[:,[self.num_actions-1]] = 1.0
        return t1

    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None, buff_act_prob=None):

        pass

    def train(self, ep=None):

        pass

    def clear_buffer(self):

        pass

    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached,mean_act_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

        # ---- Entropy
        for e in range(self.num_edges):
            for a in range(self.num_actions):
                # set_trace()
                self.writer.add_scalar('Mean_Action_Count_' + str(e)+"/"+str(a), mean_act_count[e][a], ep)

    def save_model(self, tot_reward, ep):


        pass


def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    