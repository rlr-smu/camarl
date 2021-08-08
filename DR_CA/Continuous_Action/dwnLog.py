"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 23 Jul 2019
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
import pdb
import rlcompleter
from auxLib import remoteServices, loadDataStr,  getOS
from ipdb import set_trace

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
if getOS() == "ubuntu":
    expRes = "/home/james/Dropbox/Buffer/ExpResults"
elif getOS() == "mac":
    expRes = "/Users/james/Dropbox/Buffer/ExpResults"
else:
    expRes = "/data/james_home/singularityProjects/ubuntu/ExpResults"

# =============================== Variables ================================== #

MAX_ITR = 1000000
SLEEP_TIME = 5 * 60
home = {"strider": "/home/james", "frodo": "/home/james", "zeta2": "/home/jamess", "unicen": "/home/james"}
uname_hash = {"strider": "james", "frodo": "james", "zeta2": "jamess", "unicen": "james"}


# ============================================================================ #

def dwn(server=None, rpath=None, lpath=None):
    uname = uname_hash[server]
    sr = remoteServices(server=server, uname=uname)
    rpath = home[server] + "/" + rpath
    sr.dwnDir(lpath=lpath, rpath=rpath)

def newDownload(server=None, hash=None, MAIN_FOLDER=None, PPATH=None, RPATH=None, PROJECT=None):
    for i in hash:
        tmp = hash[i].keys()


        if MAIN_FOLDER in tmp:
            os.system("mkdir " + PPATH + "/" + str(MAIN_FOLDER) + "_" + str(hash[i][MAIN_FOLDER]))
        tmp.remove(MAIN_FOLDER)
        tmpStr = ""
        for j in tmp:
            tmpStr += str(j) + "_" + str(hash[i][j]) + "_"
        tmpStr = tmpStr[:-1]
        lpath = PPATH + "/" + str(MAIN_FOLDER) + "_" + str(hash[i][MAIN_FOLDER]) + "/" + tmpStr
        os.system("mkdir " + lpath)

        rpath = RPATH + "/" + PROJECT + str(i) + "/log/*"
        if server == "unicen":
            os.system("cp -r " + rpath + " " + lpath)
        else:
            dwn(server=server, rpath=rpath, lpath=lpath)

def zeta2(server=None, sel_hash=None, MAIN_FOLDER=None, PROJECT=None, RPATH=None, LPATH=None):

    os.system("zdw.py -d " + RPATH + "/newParam_" + server + ".pkl")
    hash = loadDataStr("newParam_" + server)
    hash_new = {}
    for k1 in hash:
        flag = True
        for t1 in sel_hash:
            if sel_hash[t1] <> hash[k1][t1]:
                flag = False
        if flag:
            hash_new.update({k1:hash[k1]})
    newDownload(server=server, hash=hash_new, MAIN_FOLDER=MAIN_FOLDER, PPATH=LPATH, RPATH=RPATH, PROJECT=PROJECT)
    os.system("rm newParam_" + server + ".pkl")

def strider(sel_hash=None):

    server = "strider"
    MAP_ID = "3_1"
    LPATH = "./" + MAP_ID + "/buff_vio_true/ent_1e-4"
    MAIN_FOLDER = "WEIGHT_VARIANCE"
    RPATH = "Runs/"+MAP_ID+"_op_2"
    PROJECT = "collective_option"
    os.system("sdw.py -d " + RPATH + "/newParam_" + server + ".pkl")
    hash = loadDataStr("newParam_" + server)
    hash_new = {}
    for k1 in hash:
        flag = True
        for t1 in sel_hash:
            if sel_hash[t1] <> hash[k1][t1]:
                flag = False
        if flag:
            hash_new.update({k1:hash[k1]})
    newDownload(server=server, hash=hash_new, MAIN_FOLDER=MAIN_FOLDER, PPATH=LPATH, RPATH=RPATH, PROJECT=PROJECT)
    os.system("rm newParam_" + server + ".pkl")

def frodo(server=None, sel_hash=None, MAIN_FOLDER=None, PROJECT=None, RPATH=None, LPATH=None):

    os.system("fdw.py -d " + RPATH + "/newParam_" + server + ".pkl")
    hash = loadDataStr("newParam_" + server)

    hash_new = {}
    for k1 in hash:
        flag = True
        for t1 in sel_hash:
            if sel_hash[t1] <> hash[k1][t1]:
                flag = False
        if flag:
            hash_new.update({k1:hash[k1]})
    newDownload(server=server, hash=hash_new, MAIN_FOLDER=MAIN_FOLDER, PPATH=LPATH, RPATH=RPATH, PROJECT=PROJECT)
    os.system("rm newParam_" + server + ".pkl")

def test():

    # MAP_ID = "9_3"
    PROJECT = "bluesky_no_arr"
    MAIN_FOLDER = "LEARNING_RATE"

    # entList = [1e-3]    
    # capList = [0.2, 0.4, 0.6, 0.8]
    # seedList = [309, 332, 366, 382, 370]
    paramList = [1, 2]
    
    for param in paramList:

        os.system("mkdir "+expRes+"/Project3/Results/April_2/map2/count")
        LPATH = expRes+"/Project3/Results/April_2/map2/count"

        # RPATH = "Runs/"+"pgf_op_caps"
        RPATH = "Runs/count"

        sel_hash = {"BATCH_SIZE":param}
        frodo(server="frodo", MAIN_FOLDER=MAIN_FOLDER, sel_hash=sel_hash, RPATH=RPATH, LPATH=LPATH, PROJECT=PROJECT)
                
        # zeta2(server="zeta2", MAIN_FOLDER=MAIN_FOLDER, sel_hash=sel_hash, RPATH=RPATH, LPATH=LPATH, PROJECT=PROJECT)

def main():


    test()
    exit()


# =============================================================================== #

if __name__ == '__main__':
    main()
