"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 17 Sep 2020
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


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #
PY_ENV = "/home/james/miniconda3/envs/cop_nav/bin/python"

# ============================================================================ #

def main():

    pro_folder = os.getcwd()
    os.system(PY_ENV+" "+pro_folder+"/main.py")



# =============================================================================== #

if __name__ == '__main__':
    main()
