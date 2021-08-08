"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Mar 2020
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
from bluesky.tools import geo
from ipdb import set_trace
import numpy as np
from numpy import save
from parameters import ALT, NUM_LOS, ALPHA, BETA, FF, WIND, LOS_NBR, SEED, LARGE, HORIZON, BATCH_SIZE, MAP_ID, MAX_AC, ARR_INTERVAL, AGENT, CURR_ARR, LOCAL, MAX_WIND, HUGE, ENTROPY_WEIGHT, MAX_ALT, SLICE_INDEX, NEW_NUM_AC, ARR_RATE, ARR_RATE_INTERVAL, REAL_START_TIME, INSTANCE, ENV_AC, VMAX_VMIN, DISCOUNT, REAL_DATA_PATH, DATE, REAL_END_TIME, MIN_DISTANCE, ADD_TRAFF, RANGE_STR, SAC_EPS_DIM, SAC_ALPHA, NUM_ACTIONS, ACTION_SCALE
import networkx as nx
import matplotlib.pyplot as plt
import math
from auxLib3 import file2list, dumpDataStr, loadDataStr, listFiles
from numpy import genfromtxt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from functools import reduce
from datetime import datetime
from tqdm import tqdm
import random

random.seed(SEED)
# =============================== Variables ================================== #


# ============================================================================ #

class syn_data(object):

    def __init__(self, dir_name=None, pro_folder=None, lg = None):

        # ------ Parameters
        self.lg = lg
        self.pro_folder = pro_folder
        self.dir_name = dir_name
        self.ff = FF
        self.alt = ALT
        self.num_los = NUM_LOS
        self.los_nbr = LOS_NBR
        self.wind_flag = WIND
        self.map_id = MAP_ID
        self.local = LOCAL
        self.max_wind = MAX_WIND
        self.huge = HUGE
        self.entropy_weight = ENTROPY_WEIGHT
        self.seed = SEED
        self.large = LARGE
        self.horizon = HORIZON
        self.batch_size = BATCH_SIZE
        self.agent = AGENT
        self.max_ac = MAX_AC
        self.instance = INSTANCE

        # ------- SAC
        self.sac_eps_dim = SAC_EPS_DIM
        self.rw_scale = int(1/SAC_ALPHA)

        self.vmin = VMAX_VMIN[0]
        self.vmax = VMAX_VMIN[1]


        # -------
        self.los_hash = np.array([2, 3.5, 5, 7, 9, 10])
        self.at_rw = self.compute_reward()
        self.at_rw_mat = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                                   [-1.0, 0.075, 0.075, 0.075, 0.075, 0.075],
                                   [-1.0, 0.075, 0.15,  0.15,  0.15, 0.15],
                                   [-1.0, 0.075, 0.15,  0.25,  0.25, 0.25],
                                   [-1.0, 0.075, 0.15,  0.25,  0.35, 0.35],
                                   [-1.0, 0.075, 0.15,  0.25,  0.35, 0]])
        self.at_rw_mat = self.at_rw_mat.reshape(1, self.num_los, self.num_los)

        # ------ Actions
        self.action_scale = ACTION_SCALE
        ACTIONS = []
        if NUM_ACTIONS == 3:
            ACTIONS = [-0.5, 0, 0.5]
        elif NUM_ACTIONS == 5:
            ACTIONS = [-0.5, -0.2, 0, 0.2, 0.5]
        elif NUM_ACTIONS == 7:
            ACTIONS = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]



        self.action_space = np.array(ACTIONS)
        self.num_actions = self.action_space.shape[0]
        self.action_id_space = [ _ for _ in range(self.num_actions)]

        # ------ New aircraft time
        self.interval = np.array(ARR_INTERVAL)
        self.new_arrival_t = np.array([1])

        # ------ Aircraft Routes
        self.directions = loadDataStr(self.pro_folder+"/routes_james/"+MAP_ID+"/"+MAP_ID+"_directions")
        self.arr_rate = ARR_RATE
        if ARR_RATE == -1:
            self.arr_rate_arrival = None
        else:
            self.arr_rate_arrival = loadDataStr(self.pro_folder +"/routes_james/"+MAP_ID+"/instance/"+str(self.instance)+"/arr_rate_arrival_"+str(ARR_RATE)+"_"+str(self.max_ac))

        self.num_entry = len(self.directions.keys())
        lines_tmp = np.load(self.pro_folder+"/routes_james/"+MAP_ID+"/"+MAP_ID+"_lines.npy", allow_pickle=True)
        lines = []
        entry_gps = []
        for l in lines_tmp:
            lines.append(np.array(l))
        lines = np.array(lines_tmp)
        lat_list = []
        lon_list = []
        hdg_list = []
        for r in range(len(lines)):
            x1, y1 = lines[r][0], lines[r][1]
            if (x1, y1) not in entry_gps:
                entry_gps.append((x1, y1))

            hdg_list.append(lines[r][-1])
            lat_tmp = []
            lon_tmp = []
            wp = 0
            while wp < len(lines[r])-1:
                lat = lines[r][wp]
                lon = lines[r][wp+1]
                lat_tmp.append(lat)
                lon_tmp.append(lon)
                wp += 2
            lat_list.append(lat_tmp)
            lon_list.append(lon_tmp)
        self.routes_lat = np.array(lat_list)
        self.routes_lon = np.array(lon_list)
        self.routes_hgd = np.array(hdg_list)

        # --------------- Env Aircrafts
        nm_traj = lines_tmp.shape[0]
        lines_env_tmp = []
        count = 0

        for i in range(nm_traj):
            ln = lines_tmp[i][:-1]
            # set_trace()
            # t1 = int(ln.shape[0])
            t1 = int(len(ln))


            # ---- No point adding ac in later half
            t1 = int(t1/2)
            for j in range(0, t1-2, 2):
                x1, y1 = ln[j], ln[j+1]
                x2, y2 = ln[j+2], ln[j+3]
                # x3, y3 = (x1+x2)/2, (y1+y2)/2
                x3 = x1 + 1/3 * (x2 - x1)
                y3 = y1 + 1/3 * (y2 - y1)
                tmp1_list = [(x3, y3)]
                x3 = x1 + 2/3 * (x2 - x1)
                y3 = y1 + 2/3 * (y2 - y1)
                tmp1_list.append((x3, y3))
                for (xi, yi) in tmp1_list:
                    tmp1 = [xi, yi]
                    # for k in range(j+2, ln.shape[0]):
                    for k in range(j + 2, len(ln)):
                        tmp1.append(ln[k])
                    tmp1.append(int(lines_tmp[i][-1]))
                    flag = False
                    for ll in lines_env_tmp:
                        x1, y1 = ll[0], ll[1]
                        x2, y2 = tmp1[0], tmp1[1]
                        if x1 == x2 and y1 == y2:
                            flag = True
                            break
                    if flag == False:
                        lines_env_tmp.append(tmp1)
                        count += 1

        ll_dist = {}
        for i in range(len(lines_env_tmp)):
            ll = lines_env_tmp[i]
            d_list = []
            for (x_e, y_e) in entry_gps:
                x1, y1 = ll[0], ll[1]
                d = self.get_distance(x1, y1, x_e, y_e)
                d_list.append(d)
            ll_dist[i] = round(min(d_list), 2)


        op1 = list(ll_dist.values())
        op1.sort()
        op2 = list(ll_dist.items())
        op3 = []
        for i in range(len(op1)):
            j = op1[i]
            for (key, item) in op2:
                if j == item:
                    if key not in op3:
                        op3.append(key)
        lines_env_hash = op3.copy()


        lines_env = []
        for l in lines_env_tmp:
            lines_env.append(np.array(l))
        lines_env = np.array(lines_env_tmp)
        lat_list = []
        lon_list = []
        hdg_list = []
        for r in range(len(lines_env)):
            hdg_list.append(lines_env[r][-1])
            lat_tmp = []
            lon_tmp = []
            wp = 0
            while wp < len(lines_env[r])-1:
                lat = lines_env[r][wp]
                lon = lines_env[r][wp+1]
                lat_tmp.append(lat)
                lon_tmp.append(lon)
                wp += 2
            lat_list.append(lat_tmp)
            lon_list.append(lon_tmp)
        self.routes_lat_env = np.array(lat_list)
        self.routes_lon_env = np.array(lon_list)
        self.routes_hgd_env = np.array(hdg_list)
        self.num_env_ac = int(len(lines_env_hash)*ENV_AC)
        self.num_env_ac_list = lines_env_hash[0:self.num_env_ac]
        self.curr_ac = CURR_ARR + self.num_env_ac

        # ------ Polygons
        poly = np.load(self.pro_folder+"/routes_james/"+MAP_ID+"/"+MAP_ID+"_poly.npy", allow_pickle=True)
        self.poly = np.array(poly)
        # for i in range(self.poly.shape[0]):
        #     print(i, self.poly[i])
        # set_trace()
        self.num_edges = self.poly.shape[0]
        self.lines = np.array(lines)

        # ------ Edge Graph
        # self.G = self.edge_graph()
        # self.nbr_vec = {}
        # self.edge_ip_dim = np.zeros(self.num_edges)
        # for e in list(self.G.nodes()):
        #     nbr = list(nx.neighbors(self.G, e))
        #     nbr.append(e)
        #     self.nbr_vec[e] = [e]
        #     self.edge_ip_dim[e] = int(self.num_los * self.num_los)

        # ------- Mask used for diff_rw analy
        self.mask_diff_ana = []
        for t in range(self.horizon):
            self.mask_diff_ana.append(np.eye(self.num_edges * self.num_actions))
        self.mask_diff_ana = np.array(self.mask_diff_ana)
        self.mask_diff_ana = self.mask_diff_ana.reshape(self.horizon*self.num_edges * self.num_actions, self.num_edges * self.num_actions)

        # ----- Reward Train
        t1 = self.at_rw_mat.reshape(1, self.num_los*self.num_los)

        self.rw_train_t = t1
        self.rw_train_z = np.repeat(t1, [self.num_edges], axis=0)
        self.rw_train_z = self.rw_train_z.reshape(self.num_edges, self.num_los*self.num_los)

        self.rw_train = np.repeat(t1, [self.num_edges*self.horizon], axis=0)

        self.rw_train = self.rw_train.reshape(self.horizon, self.num_edges,  self.num_los*self.num_los)



        # ------ Discounted Return gamma
        self.gama = []
        for t in range(self.horizon):
            tmp1 = []
            for tp in range(t, self.horizon):
                tmp1.append(math.pow(DISCOUNT, (tp - t)))
            self.gama.append(np.array(tmp1))

        # ------ Epsilon
        if self.sac_eps_dim == 2:
            self.eps_bins = np.array([-1.0, 0, 1.0])

        elif self.sac_eps_dim == 4:
            self.eps_bins = np.array([-1.0, -0.5, 0, 0.5, 1.0])

        elif self.sac_eps_dim == 6:
            self.eps_bins = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 0.8, 1.0])

        if self.sac_eps_dim == 10:
            self.eps_bins = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

        elif self.sac_eps_dim == 20:
            self.eps_bins = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5,-0.4,-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        self.eps_dim = self.eps_bins.shape[0]+1

        # ------ zGraph
        self.G = self.get_zGraph()
        self.zMap = {}
        for z in list(self.G.nodes()):
            self.zMap[z] = []
            nbr = list(nx.neighbors(self.G, z))
            for zp in list(self.G.nodes()):
                if zp != z and zp not in nbr:
                    self.zMap[z].append(zp)
        self.zMap[-1] = []

        # ----- Diff Reward
        self.ntsk_indx = self.num_los * self.num_los * self.eps_dim
        self.nts_indx = self.num_los * self.num_los

        self.mid_ntsk = int(self.ntsk_indx / 2)
        self.mid_nts  = int(self.nts_indx/2)

        self.max_ntsk = self.ntsk_indx
        self.max_nts = self.nts_indx

    def display_data(self):

        self.lg.writeln("# ------------ Data ------------ #")
        self.lg.writeln("seed : " + str(self.seed))
        self.lg.writeln("map_id : "+str(self.map_id))
        self.lg.writeln("agent : "+str(self.agent))
        self.lg.writeln("pro_folder : "+str(self.pro_folder))
        self.lg.writeln("dir_name : "+str(self.dir_name))
        self.lg.writeln("num_edges : " + str(self.num_edges))
        self.lg.writeln("num_los : "+str(self.num_los))
        self.lg.writeln("los_nbr : "+str(self.los_nbr))
        self.lg.writeln("horizon : "+str(self.horizon))
        self.lg.writeln("batch_size : "+str(self.batch_size))
        self.lg.writeln("max_ac : "+str(self.max_ac))
        self.lg.writeln("at_rw : "+str(self.at_rw))
        self.lg.writeln("action_space : "+str(self.action_space))
        self.lg.writeln("num_actions : "+str(self.num_actions))
        self.lg.writeln("interval : "+str(self.interval))
        self.lg.writeln("num_entry : "+str(self.num_entry))
        self.lg.writeln("los_hash : "+str(self.los_hash))
        self.lg.writeln("poly : "+str(self.poly))
        self.lg.writeln("lines : "+str(self.lines))
        self.lg.writeln("routes_lat : "+str(self.routes_lat))
        self.lg.writeln("routes_lon : "+str(self.routes_lon))
        self.lg.writeln("routes_hgd : "+str(self.routes_hgd))
        self.lg.writeln("ff : "+str(self.ff))
        self.lg.writeln("alt : "+str(self.alt))
        self.lg.writeln("wind_flag : "+str(self.wind_flag))
        self.lg.writeln("large : "+str(self.large))
        self.lg.writeln("new_arrival_t : "+str(self.new_arrival_t))
        self.lg.writeln("Env aircrafts : "+str(self.num_env_ac))

    def get_distance(self, x1, y1, x2, y2):
        distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        return distance

    def compute_reward(self):

        at_rw = []
        for i in range(self.los_hash.shape[0]):
            los = self.los_hash[i]
            if los < 3:
                at_rw.append(-1)
            elif 3 <= los and los < 10:
                rw = -1 * ALPHA + BETA * los
                at_rw.append(rw)
            elif los >= 10:
                at_rw.append(0)
        return at_rw

    def get_zGraph(self):

        dg = nx.DiGraph()
        fl = self.pro_folder+"/routes_james/"+MAP_ID+"/raw.txt"
        lines = file2list(fl)
        for ln in lines:
            if "zgraph" in ln:
                t1 = ln.split(":")[1]
                t1 = t1.split("#")
                for ln2 in t1:
                    if "," in ln2:
                        i, j = ln2.split(",")
                        dg.add_edge(int(i), int(j))
        return dg

class real_data(object):

    def __init__(self):
        pass

class real_real_data(object):

    def __init__(self, dir_name=None, pro_folder=None, lg = None):

        # ------ Parameters

        self.lg = lg
        self.pro_folder = pro_folder
        self.dir_name = dir_name
        self.ff = FF
        self.alt = ALT
        self.num_los = NUM_LOS
        self.los_nbr = LOS_NBR
        self.wind_flag = WIND
        self.map_id = MAP_ID
        self.local = LOCAL
        self.max_wind = MAX_WIND
        self.huge = HUGE
        self.entropy_weight = ENTROPY_WEIGHT
        self.seed = SEED
        self.large = LARGE
        self.horizon = HORIZON
        self.batch_size = BATCH_SIZE
        self.agent = AGENT

        self.new_num_ac = NEW_NUM_AC
        self.real_start_time = REAL_START_TIME
        self.real_end_time = REAL_END_TIME
        self.vmin = VMAX_VMIN[0]
        self.vmax = VMAX_VMIN[1]
        self.dpath = REAL_DATA_PATH
        self.range_str = RANGE_STR
        self.num_env_ac = 0
        self.min_distance = MIN_DISTANCE

        # -------
        self.los_hash = np.array([2, 3.5, 5, 7, 9, 10])
        self.at_rw = self.compute_reward()
        self.at_rw_mat = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                                   [-1.0, 0.075, 0.075, 0.075, 0.075, 0.075],
                                   [-1.0, 0.075, 0.15,  0.15,  0.15, 0.15],
                                   [-1.0, 0.075, 0.15,  0.25,  0.25, 0.25],
                                   [-1.0, 0.075, 0.15,  0.25,  0.35, 0.35],
                                   [-1.0, 0.075, 0.15,  0.25,  0.35, 0]])
        self.at_rw_mat = self.at_rw_mat.reshape(1, self.num_los, self.num_los)
        self.action_space = np.array(ACTIONS)
        self.num_actions = self.action_space.shape[0]
        self.action_id_space = [ _ for _ in range(self.num_actions)]

        # ------ New aircraft time
        self.interval = np.array(ARR_INTERVAL)
        self.new_arrival_t = np.array([1])

        # ------ Aircraft Routes
        self.num_entry = -1
        self.add_traff = ADD_TRAFF

        # if ADD_TRAFF == 0:
        #     self.traj = loadDataStr(self.dpath+"/traj/traj_0")
        # else:
        #     self.traj = loadDataStr(self.dpath + "/traj/traj_"+str(ADD_TRAFF))
        # num_ac = 0
        # for t in self.traj.keys():
        #     num_ac += len(self.traj[t])
        # self.max_ac = num_ac
        # self.curr_ac = self.max_ac

        self.max_ac = 5000
        self.curr_ac = 5000

        # ------ Polygons
        poly = np.load(self.dpath +"/poly.npy", allow_pickle=True)
        self.poly = np.array(poly)
        self.num_edges = self.poly.shape[0]

        # ------- Mask used for diff_rw analy
        self.mask_diff_ana = []
        for t in range(self.horizon):
            self.mask_diff_ana.append(np.eye(self.num_edges * self.num_actions))
        self.mask_diff_ana = np.array(self.mask_diff_ana)
        self.mask_diff_ana = self.mask_diff_ana.reshape(self.horizon*self.num_edges * self.num_actions, self.num_edges * self.num_actions)

    def display_data(self):

        self.lg.writeln("# ------------ Data ------------ #")
        self.lg.writeln("seed : " + str(self.seed))
        self.lg.writeln("map_id : "+str(self.map_id))
        self.lg.writeln("agent : "+str(self.agent))
        self.lg.writeln("pro_folder : "+str(self.pro_folder))
        self.lg.writeln("dir_name : "+str(self.dir_name))
        self.lg.writeln("num_edges : " + str(self.num_edges))
        self.lg.writeln("num_los : "+str(self.num_los))
        self.lg.writeln("los_nbr : "+str(self.los_nbr))
        self.lg.writeln("horizon : "+str(self.horizon))
        self.lg.writeln("batch_size : "+str(self.batch_size))
        self.lg.writeln("max_ac : "+str(self.max_ac))
        self.lg.writeln("at_rw : "+str(self.at_rw))
        self.lg.writeln("action_space : "+str(self.action_space))
        self.lg.writeln("num_actions : "+str(self.num_actions))
        self.lg.writeln("interval : "+str(self.interval))
        self.lg.writeln("num_entry : "+str(self.num_entry))
        self.lg.writeln("los_hash : "+str(self.los_hash))
        # self.lg.writeln("poly : "+str(self.poly))
        # self.lg.writeln("lines_out : "+str(self.lines_in))
        # self.lg.writeln("lines_in : " + str(self.lines_out))
        # self.lg.writeln("routes_lat : "+str(self.routes_lat))
        # self.lg.writeln("routes_lon : "+str(self.routes_lon))
        # self.lg.writeln("routes_hgd : "+str(self.routes_hgd))
        self.lg.writeln("ff : "+str(self.ff))
        self.lg.writeln("alt : "+str(self.alt))
        self.lg.writeln("wind_flag : "+str(self.wind_flag))
        self.lg.writeln("large : "+str(self.large))
        self.lg.writeln("new_arrival_t : "+str(self.new_arrival_t))

    def compute_reward(self):

        at_rw = []
        for i in range(self.los_hash.shape[0]):
            los = self.los_hash[i]
            if los < 3:
                at_rw.append(-1)
            elif 3 <= los and los < 10:
                rw = -1 * ALPHA + BETA * los
                at_rw.append(rw)
            elif los >= 10:
                at_rw.append(0)
        return at_rw

    # def edge_graph(self):
    #
    #     dg = nx.DiGraph()
    #     fl = self.pro_folder+"/routes_james/edge_graph.txt"
    #     # lines = self.ax.file2list(fl)
    #     lines = file2list(fl)
    #     for ln in lines:
    #         if self.map_id in ln:
    #             t1 = ln.split(":")[1]
    #             t1 = t1.split("#")
    #             for t2 in t1:
    #                 t3 = t2.split(",")
    #                 dg.add_edge(int(t3[0]), int(t3[1]))
    #     return dg

class real_real_data_eval(object):

    def __init__(self, dir_name=None, pro_folder=None, lg = None):

        # ------ Parameters

        self.lg = lg
        self.pro_folder = pro_folder
        self.dir_name = dir_name
        self.ff = FF
        self.alt = ALT
        self.num_los = NUM_LOS
        self.los_nbr = LOS_NBR
        self.wind_flag = WIND
        self.map_id = MAP_ID
        self.local = LOCAL
        self.max_wind = MAX_WIND
        self.huge = HUGE
        self.entropy_weight = ENTROPY_WEIGHT
        self.seed = SEED
        self.large = LARGE
        self.horizon = HORIZON
        self.batch_size = BATCH_SIZE
        self.agent = AGENT
        self.date = DATE

        self.new_num_ac = NEW_NUM_AC
        self.real_start_time = REAL_START_TIME
        self.real_end_time = REAL_END_TIME
        self.vmin = VMAX_VMIN[0]
        self.vmax = VMAX_VMIN[1]
        self.dpath = REAL_DATA_PATH
        self.range_str = RANGE_STR
        self.num_env_ac = 0
        self.min_distance = MIN_DISTANCE

        # -------
        self.los_hash = np.array([2, 3.5, 5, 7, 9, 10])
        self.at_rw = self.compute_reward()
        self.at_rw_mat = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                                   [-1.0, 0.075, 0.075, 0.075, 0.075, 0.075],
                                   [-1.0, 0.075, 0.15,  0.15,  0.15, 0.15],
                                   [-1.0, 0.075, 0.15,  0.25,  0.25, 0.25],
                                   [-1.0, 0.075, 0.15,  0.25,  0.35, 0.35],
                                   [-1.0, 0.075, 0.15,  0.25,  0.35, 0]])
        self.at_rw_mat = self.at_rw_mat.reshape(1, self.num_los, self.num_los)
        self.action_space = np.array(ACTIONS)
        self.num_actions = self.action_space.shape[0]
        self.action_id_space = [ _ for _ in range(self.num_actions)]

        # ------ New aircraft time
        self.interval = np.array(ARR_INTERVAL)
        self.new_arrival_t = np.array([1])

        # ------ Aircraft Routes
        self.num_entry = -1
        self.add_traff = ADD_TRAFF

        if ADD_TRAFF == 0:
            self.traj = loadDataStr(self.dpath+"/"+str(self.date)+"_positions"+"/"+self.range_str+"/traj_eval/traj_0")
            self.arr_traj = loadDataStr(self.dpath+"/"+str(self.date)+"_positions"+"/"+self.range_str+"/traj_eval/arr_traj_0")
        else:
            self.traj = loadDataStr(self.dpath+"/"+str(self.date)+"_positions"+"/"+self.range_str + "/traj_eval/traj_"+str(ADD_TRAFF))

            self.arr_traj = loadDataStr(self.dpath+"/"+str(self.date)+"_positions"+"/"+self.range_str + "/traj_eval/arr_traj_"+str(ADD_TRAFF))

        num_ac = 0
        for t in self.arr_traj.keys():
            if self.real_start_time < t and t < self.real_end_time:
                num_ac += len(self.arr_traj[t])
        self.max_ac = num_ac
        self.curr_ac = self.max_ac


        # ------ Polygons
        poly = np.load(self.dpath +"/poly.npy", allow_pickle=True)
        self.poly = np.array(poly)
        self.num_edges = self.poly.shape[0]

        # ------- Mask used for diff_rw analy
        self.mask_diff_ana = []
        for t in range(self.horizon):
            self.mask_diff_ana.append(np.eye(self.num_edges * self.num_actions))
        self.mask_diff_ana = np.array(self.mask_diff_ana)
        self.mask_diff_ana = self.mask_diff_ana.reshape(self.horizon*self.num_edges * self.num_actions, self.num_edges * self.num_actions)

    def display_data(self):

        self.lg.writeln("# ------------ Data ------------ #")
        self.lg.writeln("seed : " + str(self.seed))
        self.lg.writeln("map_id : "+str(self.map_id))
        self.lg.writeln("agent : "+str(self.agent))
        self.lg.writeln("pro_folder : "+str(self.pro_folder))
        self.lg.writeln("dir_name : "+str(self.dir_name))
        self.lg.writeln("num_edges : " + str(self.num_edges))
        self.lg.writeln("num_los : "+str(self.num_los))
        self.lg.writeln("los_nbr : "+str(self.los_nbr))
        self.lg.writeln("horizon : "+str(self.horizon))
        self.lg.writeln("batch_size : "+str(self.batch_size))
        self.lg.writeln("max_ac : "+str(self.max_ac))
        self.lg.writeln("at_rw : "+str(self.at_rw))
        self.lg.writeln("action_space : "+str(self.action_space))
        self.lg.writeln("num_actions : "+str(self.num_actions))
        self.lg.writeln("interval : "+str(self.interval))
        self.lg.writeln("num_entry : "+str(self.num_entry))
        self.lg.writeln("los_hash : "+str(self.los_hash))
        # self.lg.writeln("poly : "+str(self.poly))
        # self.lg.writeln("lines_out : "+str(self.lines_in))
        # self.lg.writeln("lines_in : " + str(self.lines_out))
        # self.lg.writeln("routes_lat : "+str(self.routes_lat))
        # self.lg.writeln("routes_lon : "+str(self.routes_lon))
        # self.lg.writeln("routes_hgd : "+str(self.routes_hgd))
        self.lg.writeln("ff : "+str(self.ff))
        self.lg.writeln("alt : "+str(self.alt))
        self.lg.writeln("wind_flag : "+str(self.wind_flag))
        self.lg.writeln("large : "+str(self.large))
        self.lg.writeln("new_arrival_t : "+str(self.new_arrival_t))

    def compute_reward(self):

        at_rw = []
        for i in range(self.los_hash.shape[0]):
            los = self.los_hash[i]
            if los < 3:
                at_rw.append(-1)
            elif 3 <= los and los < 10:
                rw = -1 * ALPHA + BETA * los
                at_rw.append(rw)
            elif los >= 10:
                at_rw.append(0)
        return at_rw

    # def edge_graph(self):
    #
    #     dg = nx.DiGraph()
    #     fl = self.pro_folder+"/routes_james/edge_graph.txt"
    #     # lines = self.ax.file2list(fl)
    #     lines = file2list(fl)
    #     for ln in lines:
    #         if self.map_id in ln:
    #             t1 = ln.split(":")[1]
    #             t1 = t1.split("#")
    #             for t2 in t1:
    #                 t3 = t2.split(",")
    #                 dg.add_edge(int(t3[0]), int(t3[1]))
    #     return dg

class store_routes:

    def __init__(self, poly=None, lines=None, map_name=None, pro_folder=None, directions=None):
        self.poly = np.array(poly)
        self.lines = np.array(lines)
        save(pro_folder+"/routes_james/"+map_name+"/"+map_name+"_poly.npy", self.poly)
        save(pro_folder+"/routes_james/"+map_name+"/"+map_name+"_lines.npy", self.lines)
        dumpDataStr(pro_folder+"/routes_james/"+map_name+"/"+map_name+"_directions", directions)

class gen_graph:

    def __init__(self, d=1.414):

        xmin, xmax = 0, 20
        ymin, ymax = 0, 20
        pro_folder = os.getcwd()

        #lines = self.compute_line_syn(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax)

        # coor = {0: (10, 10), 1: (14, 7), 2: (16, 14), 3: (20, 10)}
        # edg_list = [(0, 1), (0, 2), (1, 3), (2, 3)]

        # lines = [[(10, 10), (14, 7)],
        #          [(10, 10), (16, 14)],
        #          [(14, 7), (20, 10)],
        #          [(16, 14), (20, 10)]]

        lines = [[(10, 10), (7, 14)],
                 [(10, 10), (14, 16)],
                 [(7, 14), (10, 20)],
                 [(14, 16), (10, 20)]]

        poly_np = []
        lines_np = []
        for line in lines:
            lines_np.append([line[0][0], line[0][1], line[1][0], line[1][1],  180])
            boundry = self.compute_poly(line=line, d=d)
            poly_np.append(boundry)

        map_name = "syn_0"
        self.poly = np.array(poly_np)
        self.lines = np.array(lines_np)
        save(pro_folder+"/routes_james/"+map_name+"_poly.npy", self.poly)
        save(pro_folder+"/routes_james/"+map_name+"_lines.npy", self.lines)

        print(np.array(lines_np))
        print(np.array(poly_np))
        exit()
        line = [(16, 14), (10, 10)]
        boundry = self.compute_poly(line=line, d=d)

        print("line", line)
        print("Boundry", boundry)

    def compute_line_syn(self, xmin=None, xmax=None, ymin=None, ymax=None, ):

        pass

    def compute_line_real(self):
        pass

    def compute_poly(self, line=None, d=None):

        poly = self.get_rectangle(line=line, d=d)
        return poly

    def get_rectangle(self, line=None, d = None):

        x1, y1 = line[0][0], line[0][1]
        x2, y2 = line[1][0], line[1][1]

        def left_up(x1, y1, x2, y2):
            l = self.distance(x1, y1, x2, y2)
            tan_theta = 1 * d/l
            x3 = x1 + tan_theta * (-(y2 - y1))
            y3 = y1 + tan_theta * (x2 - x1)
            return round(x3, 3), round(y3, 3)

        def left_dwn(x1, y1, x2, y2):
            l = self.distance(x1, y1, x2, y2)
            tan_theta = -1 * d/l
            x3 = x1 + tan_theta * (-(y2 - y1))
            y3 = y1 + tan_theta * (x2 - x1)
            return round(x3, 3), round(y3, 3)

        pt_list = []
        x3, y3 = left_up(x1, y1, x2, y2)
        pt_list.append(x3)
        pt_list.append(y3)
        x3, y3 = left_dwn(x1, y1, x2, y2)
        pt_list.append(x3)
        pt_list.append(y3)
        x3, y3 = left_up(x2, y2, x1, y1)
        pt_list.append(x3)
        pt_list.append(y3)
        x3, y3 = left_dwn(x2, y2, x1, y1)
        pt_list.append(x3)
        pt_list.append(y3)
        return np.array(pt_list)

    def distance(self, x1, y1, x2, y2):

        return abs(math.sqrt(math.pow((x2 - x1),2) + math.pow((y2 - y1),2)))
    #
    # def graph_nx(self):
    #
    #     fig, ax = plt.subplots()
    #     ax.grid("on")
    #     # coor = {0: (10, 10), 1: (14, 7), 2: (16, 14), 3: (20, 10)}
    #     # edg_list = [(0, 1), (0, 2), (1, 3), (2, 3)]
    #
    #     coor = {0: (10, 10), 1: (16, 14)}
    #     edg_list = [(0, 1)]
    #
    #     rec_hash, rec_edg = self.get_rectangle(line=coor)
    #
    #     coor.update(rec_hash)
    #     edg_list.extend(rec_edg)
    #
    #     G = nx.Graph()
    #     for n in coor.keys():
    #         G.add_node(n, pos=coor[n])
    #
    #     pos = nx.get_node_attributes(G, 'pos')
    #
    #     for a, b in edg_list:
    #         G.add_edge(a, b)
    #
    #     nx.draw(G, pos, with_labels = True, ax=ax)
    #     plt.limits = plt.axis('on')
    #     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    #     plt.show()

def extract_raw(path):

    # 4.5 long, 50 lat
    # Polygons
    pro_folder = os.getcwd()
    tmp_lines = file2list(pro_folder+"/"+path+"/raw.txt")
    poly = []
    for l in tmp_lines:
        if "pspolygon" in l:
            t1 = l.split("]")[1][1:]
            t1 = t1[:-1]
            t2 = t1.split(")(")
            t3 = []
            for p in t2:
                t4 = p.split(",")
                t3.append(float(t4[1]))
                t3.append(float(t4[0]))
            poly.append(t3)

    # Linesaaaaaa
    lines = [[51.6, 5.1, 51.6, 5.4, 51.6, 5.7, 51.6, 6.0, 51.6, 6.3, 51.6,  6.6, 51.6, 6.9, 51.6, 7.2, 51.6, 7.5, 51.6, 7.8, 51.6, 8.1, 100],
             [51.6, 5.1, 51.6, 5.4, 51.6, 5.7, 51.6, 6.0, 51.2, 6.3, 51.2, 6.6, 51.2, 6.9, 51.6, 7.2, 51.6, 7.5, 51.6, 7.8, 51.6, 8.1, 100]]


    # Directions
    # how many routes starts from a starting edges
    # directions[<start_edge>] = [<route_id>, <route_id>,...]

    directions = {}
    for l in tmp_lines:
        if "directions" in l:
            t1 = l.split(":")[1]
            t2 = t1.split("*")
            for i in t2:
                t3 = i.split("#")
                t4 = t3[1].split(",")
                t4 = list(map(lambda x : int(x), t4))
                directions[int(t3[0])] = t4

    return poly, lines, directions

def main():

    # real
    # extract_raw_real()
    #
    # exit()
    #
    pro_folder = os.getcwd()
    # map11
    mapID = "map13b"
    path = "routes_james/"+mapID
    poly, lines, directions = extract_raw(path)
    store_routes(poly=poly, lines=lines, directions=directions, map_name=mapID, pro_folder=pro_folder)


    # ----- Real dataset preprocess
    # Asia
    # countries = ["Singapore", "Malaysia", "Indonesia", "Thailand"]
    # countries = ["Singapore", "Malaysia", "Thailand"]
    # countries = ["Singapore", "Malaysia"]
    # ac_code = ['WBKT', 'WBKS', 'WBKD', 'WBKW', 'WBKK', 'WBKL', 'WBGW', 'WBGJ', 'WBGQ', 'WBGZ', 'WBGR', 'WBGM', 'WBMU', 'WBGI','WBGF','WBGL']
    # ac_code = ['WBKT', 'WBKS', 'WBKD', 'WBKW', 'WBKK', 'WBKL', 'WBGW', 'WBGQ', 'WBGZ', 'WBGR', 'WBGM']
    # Europe
    # countries = ["France", "Germany", "Netherlands"]
    # countries = ["France", "Germany"]
    # countries = ["Germany"]
    # countries = ["France"]
    # countries = ["United Kingdom"]
    # ac_code = []
    # rd = real_data()
    # rd.preprocess(countries=countries, ac_code=ac_code)


    # ----- Flight24 Dataset

    # dt.get_lines()
    # dt.get_poly()
    # dt.preprocess_all()

    pass

# =============================================================================== #

if __name__ == '__main__':
    main()
    