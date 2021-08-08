"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 08 Sep 2020
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print("# ============================ START ============================ #")
# ================================ Imports ================================ #
import sys
import os
import ipdb
import numpy as np
from parameters import NUM_AGENTS

# =============================== Variables ================================== #



# ============================================================================ #

class random_policy(object):
    def __init__(self, dim_c=None, lg=None, dir_name=None, pro_folder=None):
        self.dim_c = dim_c
        self.lg = lg
        self.dir_name = dir_name

    def action(self, obs):

        act_n = []
        act_id = []
        for i in range(NUM_AGENTS):
            u = np.zeros(5)  # 5-d because of no-move action
            a = np.random.randint(0, 5)
            u[a] = 1
            act = np.concatenate([u, np.zeros(self.dim_c)])
            act_n.append(act)
            act_id.append(a)
        return act_n, act_id



def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    print("# ============================  END  ============================ #")