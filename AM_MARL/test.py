"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 26 May 2021
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
# from utils import cosine_sim, cosine_sim_py, now, getRuntime, cosine_sim_arr
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
from environments.grid_navigation.grid_navigation import vizualize
# import matplotlib.pyplot as plt
# plt.style.use("seaborn")
from parameters import GRID

# =============================== Variables ================================== #


# ============================================================================ #

def main():

    viz = vizualize(grid=GRID)

    # viz.topology(plt)
    # exit()

    for t in range(1000):

        s = np.random.randint(10, size=(GRID*GRID))
        viz.step(s)
        print(s)






# =============================================================================== #

if __name__ == '__main__':
    main()
