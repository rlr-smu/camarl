"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 12 Aug 2020
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
from auxLib3 import file2y
import matplotlib.pyplot as plt

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

def main():

    # arr 0.4

    arr_4 = file2y(path="log_inf_0.4.txt", string="Total Reward")
    arr_5 = file2y(path="log_inf_0.5.txt", string="Total Reward")
    arr_6 = file2y(path="log_inf_0.6.txt", string="Total Reward")

    plt.plot(arr_4, label="0.4")
    plt.plot(arr_5, label="0.5")
    plt.plot(arr_6, label="0.6")
    plt.legend()
    plt.show()

    set_trace()

    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
