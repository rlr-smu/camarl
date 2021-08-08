""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, scr, tools
from parameters import EPISODES, HORIZON, UPDATE_INTV, SEED, REAL_DATA_PATH, DATE, AGENT, MAP_ID
from ipdb import set_trace
import numpy as np
from utils import log
from raw_env_real_real import raw_env_real_real
import os
import time
from auxLib3 import file2list, dumpDataStr, loadDataStr, listFiles
from numpy import genfromtxt
from datetime import datetime

class raw_data:

    def __init__(self):

        self.dpath = REAL_DATA_PATH
        cwd = self.dpath+"/uk/heathrow/5k_10k"
        fpath = cwd + "/" + DATE+"_positions_filter_uk_htr"
        ac_list = listFiles(fpath)
        for ac in ac_list:
            my_data = genfromtxt(fpath+"/"+ac, delimiter=',')
            if my_data.shape[0] == 8 and len(my_data.shape) == 1:
                my_data = np.array([my_data])
            st = int(my_data[0][0])
            st = self.get_time(st)
            en = int(my_data[-1][0])
            en = self.get_time(en)
            t = int(my_data[0][0])
            t1 = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
            t2 = t1.split(" ")[1]
            t3 = t2.split(":")
            hh = int(t3[0])
            mm = int(t3[1])
            arr_time = int(hh * 60 + mm)

            set_trace()

    def get_time(self, unix_time):

        t1 = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
        t2 = t1.split(" ")[1]
        t3 = t2.split(":")
        hh = int(t3[0])
        mm = int(t3[1])
        new_time = int(hh * 60 + mm)
        return new_time

def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name':     'extrpl',
        'plugin_type':     'sim',
        'update_interval': UPDATE_INTV,
        'update':          update
        }

    stackfunctions = {     }
    # ------- RL Environment
    global ep, lg, dir_name, pro_folder, t, cenv, buff_nm_conf
    t = 0
    dir_name = MAP_ID
    cenv = raw_env_real_real()
    pro_folder = os.getcwd()
    buff_nm_conf = np.zeros(HORIZON)

    # ----- Log
    lg = log(pro_folder + "/log/"+dir_name+"/log"+".txt")
    init()

    ep = 1
    lg.writeln("\n\n# ---------------------------------------------- #")
    lg.writeln("\nEpisode : "+str(ep))
    # --------------- #
    stack.stack('SEED ' + str(SEED))
    return config, stackfunctions

def update():

    global t, ep, buff_nm_conf, cenv

    nm_conf = cenv.step(ep=ep, t=t)
    lg.writeln("\n   t : " + str(t) + " | Conflicts : " + str(nm_conf))

    # --------- Store Buffer
    buff_nm_conf[t - 1] = nm_conf

    if t == HORIZON:
        tot_cnf = buff_nm_conf.sum()
        lg.writeln("   Total Conflicts : " + str(int(tot_cnf)))
        reset()
        return
    t += 1

def preupdate():
    pass

def reset():

    global t, ep
    ep += 1
    t = 0
    if ep == EPISODES+1:
        stack.stack('STOP')
        stack.stack('SEED '+str(SEED))
    lg.writeln("# ---------------------------------------------- #")
    lg.writeln("\nEpisode : "+str(ep))
    stack.stack('IC count.scn')


def init():

    global dir_name, pro_folder
    os.system("mkdir "+pro_folder+"/log/")
    os.system("mkdir "+pro_folder+"/log/" + dir_name)
    os.system("cp "+pro_folder+"/"+"parameters.py "+pro_folder+"/log/"+dir_name+"/")
    os.system("mkdir "+pro_folder+"/log/"+dir_name+"/plots")
    os.system("mkdir "+pro_folder+"/log/" + dir_name + "/model")
