"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 18 Jul 2020
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
from ipdb import set_trace

# =============================== Variables ================================== #


# ============================================================================ #

def main():

    tmp_str = []
    cmd = "wget --user singapore_management_university --password Mu4zAIPgX3 https://secure.flightradar24.com/singapore_management_university/"
    with open("fl_rdr.txt") as f:
        for lines in f:
            t1 = lines.split("\t")
            tmp_str.append(cmd+t1[0])            
    for x in tmp_str:        
        print(x[-32:])
        os.system(x)
    # set_trace()

if __name__ == '__main__':

    main()