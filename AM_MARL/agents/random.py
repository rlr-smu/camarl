"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 25 Feb 2021
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
import numpy as np

from agents.base_agent import baseAgent

# pdb.Pdb.complete = rlcompleter.Completer(locals())
# pdb.set_trace() 


# =============================== Variables ================================== #


# ============================================================================ #


class random_agent(baseAgent):

    def __init__(self, config=None):
        super(random_agent, self).__init__(config=config)

    def get_action(self, t, nts, buff_x):

        prob_all = np.empty((0, self.num_actions))
        x = np.zeros((1, 1 + self.s_dim  + self.o_dim))
        for _ in range(self.num_states):
            prob = np.random.uniform(0, 1, size=self.num_actions)
            prob = prob/prob.sum()
            # set_trace()
            prob_all = np.vstack((prob_all, prob))

            buff_x = np.vstack((buff_x, x))
        return buff_x, prob_all

    def log_agent(self, ep):
        pass


def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
