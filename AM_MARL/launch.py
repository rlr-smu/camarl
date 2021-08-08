"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 01 Jul 2021
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
from parameters import PY_ENV, AGENT_NAME


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

def main():

    if "mem" in AGENT_NAME and "dec" in AGENT_NAME:
        os.system(PY_ENV + " main_dec_mem.py")
        # print(PY_ENV + " main_dec_mem.py")
    elif "mem" in AGENT_NAME and "dec" not in AGENT_NAME:
        os.system(PY_ENV+" main_mem.py")
        # print(PY_ENV + " main_mem.py")
    elif "mem" not in AGENT_NAME and "dec" in AGENT_NAME:
        os.system(PY_ENV + " main_dec.py")
        # print(PY_ENV + " main_dec.py")
    elif "mem" not in AGENT_NAME and "dec" not in AGENT_NAME:
        os.system(PY_ENV + " main.py")
        # print(PY_ENV + " main.py")
    else:
        print("No main found ")
        exit()


# =============================================================================== #

if __name__ == '__main__':
    main()
