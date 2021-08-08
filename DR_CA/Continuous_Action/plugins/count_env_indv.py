"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 29 Feb 2020
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
from bluesky import stack, traf, tools
import numpy as np
from bluesky.tools.areafilter import checkInside, areas
from bluesky.tools import geo
from utils import discretize_los, discretize_los_nbr
from ipdb import set_trace
from scipy.special import softmax
from functools import reduce
from data import syn_data

# ============================================================================ #


class countAirSim_indv(syn_data):

    def __init__(self, dir_name=None, pro_folder=None):
        super(countAirSim_indv,  self).__init__(dir_name=dir_name, pro_folder=pro_folder)

        # ------- Seed
        np.random.seed(self.seed)

        # ------ Temp Variables
        self.n_e_v = np.zeros((self.num_edges, self.num_actions))
        self.n_e_l_l_v = np.zeros((self.num_edges, self.num_los,  self.num_los, self.num_actions))
        # self.n_e_v_mean = np.zeros((self.num_edges, self.num_actions))
        # self.e_ac = [[] for _ in range(self.num_edges)]
        self.e_ac = np.zeros((self.num_edges, self.num_los, self.num_los, self.curr_ac))
        self.e_ac.fill(-1)

        self.goal_time = []
        self.goal_reached = 0
        self.abs_num_ac = 0
        self.edge_ac = {}
        self.ac_edge = np.zeros(self.max_ac)
        self.ac_edge.fill(-1)
        for e in range(self.num_edges):
            self.edge_ac[e] = []

        # ------
        # self.n_np = {}
        # for t in range(self.horizon):
        #     self.n_np[t] = {}
        #     for e in range(self.num_edges):
        #         for l1 in range(self.num_los):
        #             for l2 in range(self.num_los):
        #                 for ep in range(self.num_edges):
        #                     for l1p in range(self.num_los):
        #                         for l2p in range(self.num_los):
        #                             self.n_np[t][(e,l1,l2,ep,l1p,l2p)] = 0

        # self.n_np = np.zeros((self.horizon, self.num_edges, self.num_los,self.num_los,self.num_actions, self.num_edges, self.num_los,self.num_los))


    def init_env(self, route_keeper):

        # ------ Speed Up Simulation
        if self.ff:
            stack.stack('FF')

        # ------ Add Edges
        for p in range(self.poly.shape[0]):
            t1 = reduce(lambda x1, x2: " "+str(x1) + ","+str(x2), self.poly[p])
            stack.stack('POLY '+str(p)+t1)
        self.routes_list = areas.keys()

        # ------ Reset new arrival time
        self.new_arrival_t = np.zeros(self.num_entry)
        self.new_arrival_t.fill(0)

        self.n_np = np.zeros((self.horizon, self.num_edges, self.num_los,self.num_los,self.num_actions, self.num_edges, self.num_los,self.num_los))

        self.ac_state = {}
        self.train_data = {}
        for t in range(self.horizon+1):
            self.ac_state[t] = {}
            self.train_data[t] = {}


        return route_keeper

    def init_state(self):

        self.e_ac.fill(-1)
        state = np.zeros((self.num_edges, self.num_los, self.num_los))
        num_ac = traf.ntraf
        self.nt_los = self.get_distance_matrix()

        for i in range(num_ac):
            x, y = traf.lat[i], traf.lon[i]
            e = int(self.get_edges(x, y, self.alt))
            idx = traf.id[i]
            ac_id = int(idx[2:])
            self.edge_ac[e].append(ac_id)
            self.ac_edge[ac_id] = e
            los = tuple(self.nt_los[i])

            if len(los) == 1:
                state[e][los[0]][los[0]] += 1
                self.e_ac[e][los[0]][los[0]][i]  = int(idx[2:])
                # self.ac_state[0][ac_id] = (e, los[0], los[0])
            else:
                state[e][los] += 1
                self.e_ac[e][los][i] = int(idx[2:])
                # self.ac_state[0][ac_id] = (e, los[0], los[1])

        return state

    def add_wind(self):
        sp = np.random.randint(0, self.max_wind)
        stack.stack('WIND 52.29145730495272, 4.777671441950008,2000,90,' + str(sp))

    def step(self, ep=None,  t=None, state=None, action_prob=None, route_keeper=None, action_ac=None, agent=None):

        self.ac_speed = {}
        store_terminal = np.zeros(len(traf.id), dtype=int)
        # ----- Reward
        rt = self.get_reward(state)
        # print(action_prob)
        for e in range(self.num_edges):
            for l1 in range(self.num_los):
                for l2 in range(self.num_los):
                    n_e_l_l = state[e][l1][l2]
                    pBucket = action_prob[e]
                    pBucket = pBucket/pBucket.sum()
                    # print(pBucket)
                    self.n_e_l_l_v[e][l1][l2] = np.random.multinomial(n_e_l_l, pBucket)

        # ----- Action
        if agent == "ppo":
            for i in range(len(traf.id)):
                speed = action_ac[i]
                stack.stack('{} SPD  {}'.format(traf.id[i], speed))
        elif "mtmf" in agent:
            for i in range(len(traf.id)):
                speed = action_ac[i]
                stack.stack('{} SPD  {}'.format(traf.id[i], speed))
        else:
            self.apply_action(t, self.n_e_l_l_v)

        # ----- Add wind
        if self.wind_flag:
            self.add_wind()

        # ----- Add Traffic
        if self.arr_rate == -1:
            route_keeper = self.add_traffic(t=t,route_keeper=route_keeper)
        else:
            route_keeper = self.add_traffic_new(t=t,route_keeper=route_keeper)

        # ----- Compute perf metrics
        self.nt_los = self.get_distance_matrix()
        conflict = np.sort(self.dist_mat, 1)[:,1:2]
        conflict = np.where(conflict < 3, 1, 0)
        # num_confl = int(conflict.sum()/2)
        num_confl = int(conflict.sum())

        # ----- Next state
        self.edge_ac = {}
        self.edge_ac[-1] = []
        for e in range(self.num_edges):
            self.edge_ac[e] = []
        self.e_ac.fill(-1)
        num_ac = traf.ntraf
        next_state = np.zeros((self.num_edges, self.num_los, self.num_los))

        for i in range(num_ac):
            x, y = traf.lat[i], traf.lon[i]
            e = int(self.get_edges(x, y, self.alt))
            idx = traf.id[i]
            ac_id = int(idx[2:])
            # print(self.edge_ac)
            self.edge_ac[e].append(ac_id)
            self.ac_edge[ac_id] = e

            if e == -1:
                self.goal_time.append(traf.timeflown[i])
                self.goal_reached += 1
                store_terminal[i] = 1
                stack.stack('DEL '+idx)
            else:
                los = tuple(self.nt_los[i])
                if len(los) == 1:
                    next_state[e][los[0]][los[0]] += 1
                    self.e_ac[e][los[0]][los[0]][i] = int(idx[2:])
                    self.ac_state[t+1][ac_id] = (e, los[0], los[0])
                else:
                    next_state[e][los] += 1
                    self.e_ac[e][los][i] = int(idx[2:])
                    self.ac_state[t+1][ac_id] = (e, los[0], los[1])



                if ac_id in self.ac_state[t] and ac_id in self.ac_speed:
                    s = self.ac_state[t][ac_id]
                    sp = self.ac_state[t+1][ac_id]
                    v = self.ac_speed[ac_id]
                    indx = (s[0], s[1], s[2], v, sp[0], sp[1], sp[2])
                    indx_s = (s[0], s[1], s[2], v)
                    indx_sp = (sp[0], sp[1], sp[2])
                    if indx_s in self.train_data[t]:
                        self.train_data[t][indx_s].append(indx_sp)
                    else:
                        self.train_data[t][indx_s] = [indx_sp]

                    self.n_np[t][indx] += 1


                    # if indx == (1, 4, 5, 3, 1, 4, 5) and ep == 2 and t == 28:
                    #     print(t, self.n_np[t][indx])
                    #     set_trace()


        return next_state, self.n_e_l_l_v, rt, num_confl, store_terminal, route_keeper

    def apply_action(self, t, nellv):

        for e in range(self.num_edges):
            for l1 in range(self.num_los):
                for l2 in range(self.num_los):
                    ell = self.e_ac[e][l1][l2]
                    ac_list = ell[ell != -1]
                    st = 0

                    for v in range(self.num_actions):
                        n = nellv[e][l1][l2][v]
                        en = int(st + n)
                        ac_list2 = ac_list[st:en]
                        for id in ac_list2:

                            self.ac_state[t][int(id)] = (e, l1, l2)


                            # if v == 1:
                            #     # print(id)
                            #     index = traf.id2idx(idx)
                            #     speed = int(np.round((traf.cas[int(index)] / tools.geo.nm) * 3600))
                            # else:
                            #     speed = self.action_space[v]
                            # speed = self.action_space[v]

                            idx = 'KL' + str(int(id))
                            index = traf.id2idx(idx)
                            speed = int(np.round((traf.cas[index] / tools.geo.nm) * 3600))
                            speed_change = self.action_space[v]
                            new_speed = speed + speed * speed_change
                            if new_speed < self.vmin:
                                new_speed = self.vmin
                            elif new_speed > self.vmax:
                                new_speed = self.vmax

                            self.ac_speed[int(id)] = v
                            stack.stack('{} SPD  {}'.format(idx, new_speed))
                            # print('{} SPD  {}'.format(idx, speed))
                        st = en

    def get_reward(self, state):

        # set_trace()
        # op1 = np.tile(self.at_rw, (self.num_edges, self.num_los, 1))
        # op2 = np.multiply(op1, state)
        op1 = self.at_rw_mat.reshape(1, self.num_los, self.num_los)
        op2 = op1 * state
        op3 = op2.sum(2).sum(1)
        return op3

    def get_edges(self, x, y, alt):

        for r in self.routes_list:
            if checkInside(r, x, y, alt):
                return r
        return -1

    def add_traffic_new(self, t=None, route_keeper=None):

        for did in self.directions:
            if t in self.arr_rate_arrival[did]:
                rid = np.random.choice(self.directions[did])
                route_keeper = self.add_new_aircraft(rid, route_keeper)
        return route_keeper

    def add_traffic(self, t=None, route_keeper=None):

        t1 = np.where(self.new_arrival_t == t)[0]
        if t == 0:
            for did in range(self.num_entry):
                rid = np.random.choice(self.directions[did])
                route_keeper = self.add_new_aircraft(rid, route_keeper)
                next_ac_t = np.random.choice(self.interval)
                self.new_arrival_t[did] += next_ac_t
        else:
            for d_id in t1:
                r_id = np.random.choice(self.directions[d_id])
                route_keeper = self.add_new_aircraft(r_id, route_keeper)
                next_ac_t = np.random.choice(self.interval)
                self.new_arrival_t[d_id] += next_ac_t
        return route_keeper

    def add_new_aircraft(self, rid, route_keeper):

        if self.abs_num_ac > self.max_ac-1:
            return route_keeper

        # set_trace()

        lats = self.routes_lat[rid]
        lons = self.routes_lon[rid]

        ac_id = self.abs_num_ac
        route_keeper[ac_id] = rid
        self.abs_num_ac += 1

        # ----- Origin
        o_lat = lats[0]
        o_lon = lons[0]
        hdg = self.routes_hgd[rid]

        stack.stack('CRE KL{}, A320, {}, {}, {}, 2000, 100'.format(ac_id, o_lat, o_lon, hdg))

        # ----- Waypoints
        for i in range(1, len(lats) - 1):
            w_lat = lats[i]
            w_lon = lons[i]
            stack.stack('ADDWPT KL{}, {}, {}'.format(ac_id, w_lat, w_lon))

        # ----- Destination
        d_lat = lats[-1]
        d_lon = lons[-1]
        stack.stack('DEST KL{}, {}, {}'.format(ac_id, d_lat, d_lon))
        stack.stack('trail on')

        return route_keeper

    def get_distance_matrix(self):

        lat_mat = traf.lat
        lon_mat = traf.lon
        num_ac = traf.ntraf

        # ----- Only Distance
        self.dist_mat = geo.latlondist_matrix(np.repeat(lat_mat, num_ac), np.repeat(lon_mat, num_ac), np.tile(lat_mat, num_ac), np.tile(lon_mat, num_ac)).reshape(num_ac, num_ac)



        # ----- Distance + Direction
        # dist_mat2 = geo.qdrdist_matrix(np.repeat(lat_mat, num_ac), np.repeat(lon_mat, num_ac), np.tile(lat_mat, num_ac), np.tile(lon_mat, num_ac))

        # ----- Distance + Direction, diff way to compute
        #  Bit faster but need to subtract from 360 deg
        # dist_mat3 = geo.kwikqdrdist_matrix(np.repeat(lat_mat, num_ac), np.repeat(lon_mat, num_ac), np.tile(lat_mat, num_ac), np.tile(lon_mat, num_ac))


        # ------ Remove distances in reverse direction
        # t1 = dist_mat2[0].reshape(num_ac, num_ac)
        # t2 = dist_mat2[1].reshape(num_ac, num_ac)
        # t3 = np.where(t1 <= 0, 0, t1)
        # t4 = np.where(t3 > 0, 1, t3)
        # dist = np.multiply(t2, t4)



        ac_los = discretize_los_nbr(self.dist_mat)
        return ac_los

def main():
    print("Hello World")


'''

Poly 1 = [41.7, -92.5, 41.7, -93.8, 41.3, -93.8), (41.3, -92.5)]

Poly2 = [(41.7, -93.8), (41.7, -95), (41.3, -95), (41.3, -93.8)]

Poly3 = [(41.2, -92.5), (41.2, -93.8), (40.8, -93.8), (40.8, -92.5)]

Poly4 = [(41.2, -93.8), (41.2, -95), (40.8, -95) ,(40.8, -93.8)]

Poly5 = [(42.2, -93.6), (42.2, -94), (41.5, -94), (41.5, -93.6)]

Poly6 = [(41.5, -93.6), (41.5, -94), (41, -94), (41, -93.6)]

Poly7 = [(41, -94), (41, -93.6), (40.2, -94), (40.2, -93.6) ]

Line8 = [(41.5, -95), (41.5, -92.5)]
Line9 = [(41, -92.5), (41, -95)]
Line10 = [(42.2, -93.8), (40.2, -93.8)]

'''

# =============================================================================== #

if __name__ == '__main__':
    main()

