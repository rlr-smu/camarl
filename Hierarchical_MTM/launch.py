"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 26 Apr 2019
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import os
from parameters import AGENT, LOAD_MODEL


# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

def main():

    if LOAD_MODEL:
        if AGENT == "op_sep_ind":
            os.system("python eval_load.py -a " + AGENT)
        elif AGENT == "op_ind":
            os.system("python eval_load.py -a " + AGENT)
        elif AGENT == "op_ps_ind":
            os.system("python eval_load.py -a " + AGENT)
        elif AGENT == "pg_fict_dcp":
            os.system("python eval_load_baseline.py -a " + AGENT)
        elif AGENT == "tmin":
            os.system("python eval_load_baseline.py -a " + AGENT)
        else:
            print ("Error: Agent not recognized !")
    else:
        if AGENT == "op_ind":
            os.system("python main_option.py -a " + AGENT)
        elif AGENT == "op_pgv":
            os.system("python main_option.py -a " + AGENT)
        elif AGENT == "pg_fict_dcp":
            os.system("python main_baseline.py -a " + AGENT)
        elif AGENT == "tmin":
            os.system("python main_baseline.py -a " + AGENT)
        else:
            print ("Error: Agent not recognized !")


# =============================================================================== #

if __name__ == '__main__':
    main()
    