"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 25 Feb 2021
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
import shutil
from parameters import DISCOUNT, HORIZON, NB_DIM, DIST_THRESHOLD, NEAREST_NBR, TINY, TOTAL_AGENTS
import pickle
import torch as tc
import numba as nb
import numpy as np
from numpy import dot
from numpy.linalg import norm


# pdb.Pdb.complete = rlcompleter.Completer(locals())
# pdb.set_trace() 


# =============================== Variables ================================== #
NB_DTYPE_F = nb.types.float64
NB_DTYPE_I = nb.types.int64

# ============================================================================ #

class logRunTime:

    def __init__(self, init_time=None):

        self.internal_time = init_time

    def now(self):

        return time.time()

    def logTime(self):

        runtime = self.getRuntime(self.internal_time, self.now())
        self.internal_time = self.now()
        return str(round(runtime, 3))+" sec"

    def getRuntime(self, st, en):

        return round(en - st, 3)

def loadDataStr(x):

    file2 = open(x+'.pkl', 'rb')
    ds = pickle.load(file2)
    file2.close()
    return ds

def dumpDataStr(x,obj):

    afile = open(x+'.pkl', 'wb')
    pickle.dump(obj, afile)
    afile.close()

def deleteDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)

def get_one_hot(n_classes):
    target = tc.tensor([[ _ for _ in range(n_classes)]])
    y = tc.zeros(n_classes, n_classes).type(tc.int)
    y[range(y.shape[0]), target] = 1
    y = y.data.numpy()
    return y

def cosine_sim_py(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

@nb.njit(fastmath=True, cache=True, nogil=True)
def cosine_sim(a, b):

    if norm(a) == 0 and norm(b) == 0:
        return 1
    if norm(a) == 0:
        return 0
    if norm(b) == 0:
        return 0
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

@nb.njit(fastmath=True, cache=True, nogil=True)
def cosine_sim_arr_njit(arr1, arr2):
    arr1_size = arr1.shape[0]
    res = np.zeros(arr1_size, dtype=NB_DTYPE_F)
    for i in range(arr1_size):
        a = arr1[i]
        b = arr2[i]
        res[i] = cosine_sim(a, b)
    return res

@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])], '(n),(n)->(n)')
def cosine_sim_arr(arr1, arr2, res):
    cos_sim1 = dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    res[0] = cos_sim1

@nb.njit(fastmath=True, cache=True, nogil=True)
def similarity_measure(arr1, arr2):
    arr1_size = arr1.shape[0]
    res = np.zeros(arr1_size, dtype=NB_DTYPE_F)
    for i in range(arr1_size):
        s1 = arr1[i][0]
        o1 = arr1[i][1:]

        s2 = arr2[i][0]
        o2 = arr2[i][1:]

        if s1 == s2:
            res[i] = cosine_sim(o1, o2)
        else:
            res[i] = -1
    return res

@nb.njit(fastmath=True, cache=True, nogil=True)
def similarity_measure_act(arr1, arr2):
    arr1_size = arr1.shape[0]
    res = np.zeros(arr1_size)
    for i in range(arr1_size):
        s1 = arr1[i][0]
        a1 = arr1[i][-1]
        o1 = arr1[i][1:-1]

        s2 = arr2[i][0]
        a2 = arr2[i][-1]
        o2 = arr2[i][1:-1]

        if s1 == s2 and a1 == a2:
            res[i] = cosine_sim(o1, o2)
        else:
            res[i] = -1
    return res

@nb.njit(fastmath=True, cache=True, nogil=True)
def hash_map_act(arr_list, arr):

    l1 = arr_list
    l2 = arr.repeat(l1.shape[0]).reshape(-1, l1.shape[0]).transpose()
    res = similarity_measure_act(l1, l2)
    l3 = np.where(res > DIST_THRESHOLD, res, 0)

    if l3.sum() > 0:
        sim_indx = int(np.argmax(l3))
        return sim_indx
    else:
        return -1

@nb.njit(fastmath=True, cache=True, nogil=True)
def hash_map_act_nbr(arr_list, arr):

    l1 = arr_list
    l2 = arr.repeat(l1.shape[0]).reshape(-1, l1.shape[0]).transpose()
    res = similarity_measure_act(l1, l2)
    l3 = np.where(res > DIST_THRESHOLD, res, 0)

    sim_indx = int(-1)
    if l3.sum() > 0:
        sim_indx = int(np.argmax(l3))
        nbr_index = np.zeros(NEAREST_NBR, dtype=np.int64)
        # return sim_indx, nbr_index
    else:
        # --- sort the distance
        dist = res.copy()
        dist_index = dist.argsort()
        dist_size = dist_index.shape[0]
        nbr_index = dist_index[dist_size - NEAREST_NBR:]
    return sim_indx, nbr_index


@nb.njit(fastmath=True, cache=True, nogil=True)
def hash_map_nbr(arr_list, arr):

    l1 = arr_list
    l2 = arr.repeat(l1.shape[0]).reshape(-1, l1.shape[0]).transpose()
    res = similarity_measure(l1, l2)
    l3 = np.where(res > DIST_THRESHOLD, res, 0)

    sim_indx = int(-1)
    if l3.sum() > 0:
        sim_indx = int(np.argmax(l3))
        nbr_index = np.zeros(NEAREST_NBR, dtype=np.int64)
        # return sim_indx, nbr_index
    else:
        # --- sort the distance
        dist = res.copy()
        dist_index = dist.argsort()
        dist_size = dist_index.shape[0]
        nbr_index = dist_index[dist_size - NEAREST_NBR:]

    return sim_indx, nbr_index

@nb.njit(fastmath=True, cache=True, nogil=True)
def hash_map(arr_list, arr):

    l1 = arr_list
    l2 = arr.repeat(l1.shape[0]).reshape(-1, l1.shape[0]).transpose()
    res = similarity_measure(l1, l2)
    l3 = np.where(res > DIST_THRESHOLD, res, 0)
    if l3.sum() > 0:
        sim_indx = int(np.argmax(l3))
        return sim_indx
    else:
        # --- sort the distance
        return -1

@nb.njit(fastmath=True, cache=True, nogil=True)
def update_mem_graph(at_rew, buff_x, ntsas, ntsa, nts, s_dim, a_dim, o_dim, x_tab_old, t_tab_old, xxp_tab_old, rg_tab, fg_val, o_nbr, fg_index, fg_soa, val_fn):

    # ----
    num_action = ntsa.shape[2]

    # --------- Copy Prev Data
    x_tab = np.empty((0, s_dim + o_dim + a_dim))

    x_id = x_tab_old.shape[0] - 1
    x_tab = np.vstack((x_tab, x_tab_old))
    node_len = x_tab.shape[0]

    xxp_tab = np.empty((0, NB_DIM))
    xxp_tab = np.vstack((xxp_tab, xxp_tab_old))

    t_tab = np.empty((0, 1))
    t_tab = np.vstack((t_tab, t_tab_old))

    x_node_id = x_tab.shape[0]
    # --------- Compute New Nodes
    obs = nts
    t1 = np.where(ntsas > 0)
    t2 = np.vstack(t1).transpose()
    for i in range(t2.shape[0]):
        t = t2[i][0]
        s = t2[i][1]
        a = t2[i][2]
        sp = t2[i][3]

        # --obs
        o = np.zeros(o_dim)
        o_nbr_indexes = o_nbr[s][o_nbr[s] > -1]
        o_nbr_len = o_nbr_indexes.shape[0]
        o1 = obs[t][o_nbr_indexes]
        o[0:o_nbr_len] = o1

        # ------ node x
        tot_len = s_dim+o_dim+a_dim #o.shape[0]
        soa = np.zeros((1, tot_len))
        soa[0][0:s_dim] = s
        soa[0][s_dim:s_dim+o_dim] = o
        soa[0][s_dim+o_dim:tot_len] = a

        # ----- Check similar nodes
        x_tab_size = x_tab.shape[0]
        if x_tab_size > 0:

            # --- find node in graph
            sim_indx = hash_map_act(x_tab, soa)
            if sim_indx != -1:

                # ------ Node found in graph
                # ----- Update t
                sim_t_tab = t_tab[sim_indx][0]
                if sim_t_tab < t:
                    t_tab[sim_indx][0] = t

                # ---- Update values
                idv_val = val_fn[t][s]
                fg_val[sim_indx] = np.maximum(fg_val[sim_indx], idv_val)
                # Atomic reward multiplied with count
                rew_count = ntsa[t][s] * at_rew[t][s]
                rg_tab[sim_indx] = np.maximum(rg_tab[sim_indx], rew_count)

                # ---- fg_soa
                fg_soa[sim_indx] = np.maximum(fg_soa[sim_indx], idv_val)
                x_node_id = sim_indx

                # --- find node in buff_x, fg_index
                l1 = buff_x[:, 1:]
                so = soa[:, 0:s_dim + o_dim]
                sim_indx_idv = hash_map(l1, so)
                if sim_indx_idv != -1:
                    fg_index[sim_indx_idv] = int(sim_indx)

            else:

                # ------ Node not found in graph
                # --- Create
                x_tab = np.vstack((x_tab, soa))
                t_tab = np.vstack((t_tab, np.array([[t]])))

                # -- Update Values
                idv_val = val_fn[t][s].reshape(1, num_action)
                fg_val = np.vstack((fg_val, idv_val))
                fg_soa = np.vstack((fg_soa, idv_val))

                # Atomic reward multiplied with count
                rew_count = ntsa[t][s] * at_rew[t][s]
                rew_count = rew_count.reshape(1, num_action)
                rg_tab = np.vstack((rg_tab, rew_count))

                # --- find node in buff_x, fg_index
                l1 = buff_x[:, 1:]
                so = soa[:, 0:s_dim + o_dim]
                sim_indx_idv = hash_map(l1, so)
                if sim_indx_idv != -1:
                    fg_index[sim_indx_idv] = int(node_len)

                # ---- Update node counter
                node_len += 1
                x_id += 1
                x_node_id = x_id

                # --- create nbr list
                x_xp = np.zeros((1, NB_DIM))
                x_xp.fill(-1)
                xxp_tab = np.vstack((xxp_tab, x_xp))

        else:
            # The new node not found in the graph. Create the new node.
            # --- Create
            x_tab = np.vstack((x_tab, soa))
            t_tab = np.vstack((t_tab, np.array([[t]])))

            # -- Update Values
            idv_val = val_fn[t][s].reshape(1, num_action)
            fg_val = np.vstack((fg_val, idv_val))
            fg_soa = np.vstack((fg_soa, idv_val))

            # Atomic reward multiplied with count
            rew_count = ntsa[t][s] * at_rew[t][s]
            rew_count = rew_count.reshape(1, num_action)
            rg_tab = np.vstack((rg_tab, rew_count))

            # --- find node in buff_x, fg_index
            l1 = buff_x[:, 1:]
            so = soa[:, 0:s_dim+o_dim]
            # nbr_index = np.zeros(NEAREST_NBR)
            # sim_indx_idv, nbr_index = hash_map(l1, so, nbr_index)
            sim_indx_idv = hash_map(l1, so)
            if sim_indx_idv != -1:
                fg_index[sim_indx_idv] = int(node_len)

            # ---- Update node counter
            node_len += 1
            x_id += 1
            x_node_id = x_id

            # --- create nbr list
            x_xp = np.zeros((1, NB_DIM))
            x_xp.fill(-1)
            xxp_tab = np.vstack((xxp_tab, x_xp))


        # ------- node x'
        if t+1 < HORIZON:
            t3 = np.where(ntsa[t+1][sp] > 0)
            sp_ap = np.vstack(t3)[0]
            for ii in range(sp_ap.shape[0]):
                ap = sp_ap[ii]

                # ---- obs
                op = np.zeros(o_dim)
                o_nbr_indexes = o_nbr[sp][o_nbr[sp] > -1]
                o_nbr_len = o_nbr_indexes.shape[0]
                o1 = obs[t+1][o_nbr_indexes]
                op[0:o_nbr_len] = o1
                tot_len = s_dim + o_dim + a_dim #op.shape[0]
                sp_op_ap = np.zeros((1, tot_len))
                sp_op_ap[0][0:s_dim] = sp
                sp_op_ap[0][s_dim:s_dim+o_dim] = op
                sp_op_ap[0][s_dim+o_dim:tot_len] = ap

                # ---- find node in graph
                sim_indx = hash_map_act(x_tab, sp_op_ap)
                if sim_indx != x_node_id:
                    if sim_indx != -1:
                        # ------ Node found in graph
                        # --- update t
                        sim_t_tab = t_tab[sim_indx][0]
                        if sim_t_tab < t+1:
                            t_tab[sim_indx][0] = t+1

                        # ---- Update values
                        idv_val = val_fn[t+1][sp]
                        fg_val[sim_indx] = np.maximum(fg_val[sim_indx], idv_val)
                        # Atomic reward multiplied with count
                        rew_count = ntsa[t + 1][sp] * at_rew[t + 1][sp]
                        rg_tab[sim_indx] = np.maximum(rg_tab[sim_indx], rew_count)
                        # ---- fg_soa
                        fg_soa[sim_indx] = np.maximum(fg_soa[sim_indx], idv_val)

                        # --- find node in buff_x, fg_index
                        l1 = buff_x[:, 1:]
                        sp_op = sp_op_ap[:, 0:s_dim + o_dim]
                        sim_indx_idv = hash_map(l1, sp_op)
                        if sim_indx_idv != -1:
                            fg_index[sim_indx_idv] = int(sim_indx)

                        # ---- Update nbr list
                        if sim_indx not in xxp_tab[x_node_id] and x_node_id != sim_indx:
                            ind1_list = np.where(xxp_tab[x_node_id] == -1)[0]
                            if ind1_list.shape[0] > 0:
                                ind1 = ind1_list.min()
                                xxp_tab[x_node_id][ind1] = sim_indx
                    else:
                        # ------ Node not found in graph
                        # --- Create
                        x_tab = np.vstack((x_tab, sp_op_ap))
                        t_tab = np.vstack((t_tab, np.array([[t + 1]])))

                        # --- Update values
                        idv_val = val_fn[t+1][sp].reshape(1, num_action)
                        fg_val = np.vstack((fg_val, idv_val))
                        fg_soa = np.vstack((fg_soa, idv_val))

                        # Atomic reward multiplied with count
                        rew_count = ntsa[t + 1][sp] * at_rew[t + 1][sp]
                        rew_count = rew_count.reshape(1, num_action)
                        rg_tab = np.vstack((rg_tab, rew_count))


                        # --- find node in buff_x, fg_index
                        l1 = buff_x[:, 1:]
                        sp_op = sp_op_ap[:, 0:s_dim+o_dim]
                        sim_indx_idv = hash_map(l1, sp_op)
                        if sim_indx_idv != -1:
                            fg_index[sim_indx_idv] = int(node_len)

                        # ---- Update node counter
                        node_len += 1
                        x_id += 1

                        # --- create nbr list
                        x_xp = np.zeros((1, NB_DIM))
                        x_xp.fill(-1)
                        xxp_tab = np.vstack((xxp_tab, x_xp))

                        # ---- Update nbr list for soa
                        if x_id not in xxp_tab[x_node_id]:
                            ind1_list = np.where(xxp_tab[x_node_id] == -1)[0]
                            if ind1_list.shape[0] > 0:
                                ind1 = ind1_list.min()
                                xxp_tab[x_node_id][ind1] = x_id


    # ------- sort time index
    t_sorted = np.zeros(t_tab.shape[0])
    t_sorted.fill(-1)
    counter = 0
    for t in range(HORIZON-1, -1, -1):
        t1 = np.where(t_tab == t)[0]
        if t1.shape[0] > 0:
            t_sorted[counter:counter+t1.shape[0]] = t1
            counter += t1.shape[0]

    return x_tab, t_tab, xxp_tab, rg_tab, fg_val, t_sorted, fg_index, fg_soa

@nb.njit(fastmath=True, cache=True, nogil=True)
def update_mem_graph_dec(at_rew, ntsas, ntsa, nts, s_dim, a_dim, o_dim, x_tab_old, t_tab_old, xxp_tab_old, rg_tab, fg_val, o_nbr, val_fn, fg_soa):

    # ----
    num_action = ntsa.shape[2]

    # --------- Copy Prev Data
    x_tab = np.empty((0, s_dim + o_dim + a_dim))

    x_id = x_tab_old.shape[0] - 1
    x_tab = np.vstack((x_tab, x_tab_old))
    node_len = x_tab.shape[0]

    xxp_tab = np.empty((0, NB_DIM))
    xxp_tab = np.vstack((xxp_tab, xxp_tab_old))

    t_tab = np.empty((0, 1))
    t_tab = np.vstack((t_tab, t_tab_old))

    # x_node_id = x_tab.shape[0]
    # --------- Compute New Nodes
    obs = nts
    t1 = np.where(ntsas > 0)
    t2 = np.vstack(t1).transpose()
    for i in range(t2.shape[0]):
        t = t2[i][0]
        s = t2[i][1]
        a = t2[i][2]
        sp = t2[i][3]

        # --obs
        o = np.zeros(o_dim)
        o_nbr_indexes = o_nbr[s][o_nbr[s] > -1]
        o_nbr_len = o_nbr_indexes.shape[0]
        o1 = obs[t][o_nbr_indexes]
        o[0:o_nbr_len] = o1

        # ------ node x
        tot_len = s_dim+o_dim+a_dim #o.shape[0]
        soa = np.zeros((1, tot_len))
        soa[0][0:s_dim] = s
        soa[0][s_dim:s_dim+o_dim] = o
        soa[0][s_dim+o_dim:tot_len] = a

        # ----- Check similar nodes
        x_tab_size = x_tab.shape[0]
        if x_tab_size > 0:
            # --- find node in graph
            sim_indx = hash_map_act(x_tab, soa)
            if sim_indx != -1:

                # ------ Node found in graph
                # ----- Update t
                sim_t_tab = t_tab[sim_indx][0]
                if sim_t_tab < t:
                    t_tab[sim_indx][0] = t

                # ---- Update values
                idv_val = val_fn[t][s][a]
                fg_val[sim_indx] = max(fg_val[sim_indx][0], idv_val)

                # ---- fg_soa
                fg_soa[sim_indx] = np.maximum(fg_soa[sim_indx], val_fn[t][s])


                # --- Atomic reward multiplied with count
                rew_count = ntsa[t][s][a] * at_rew[t][s][a]
                rg_tab[sim_indx] = max(rg_tab[sim_indx][0], rew_count)
                x_node_id = sim_indx

            else:

                # ------ Node not found in graph
                # --- Create
                x_tab = np.vstack((x_tab, soa))
                t_tab = np.vstack((t_tab, np.array([[t]])))

                # --- Update Values
                idv_val = val_fn[t][s][a]
                fg_val = np.vstack((fg_val, np.array([[idv_val]])))
                fg_soa = np.vstack((fg_soa, val_fn[t][s].reshape(1, num_action)))

                # --- Atomic reward multiplied with count
                rew_count = ntsa[t][s][a] * at_rew[t][s][a]
                rg_tab = np.vstack((rg_tab, np.array([[rew_count]])))

                # ---- Update node counter
                node_len += 1
                x_id += 1
                x_node_id = x_id

                # --- create nbr list
                x_xp = np.zeros((1, NB_DIM))
                x_xp.fill(-1)
                xxp_tab = np.vstack((xxp_tab, x_xp))

        else:
            # The new node not found in the graph. Create the new node.
            # --- Create
            x_tab = np.vstack((x_tab, soa))
            t_tab = np.vstack((t_tab, np.array([[t]])))

            # --- Update Values
            idv_val = val_fn[t][s][a]
            fg_val = np.vstack((fg_val, np.array([[idv_val]])))
            fg_soa = np.vstack((fg_soa, val_fn[t][s].reshape(1, num_action)))


            # --- Atomic reward multiplied with count
            rew_count = ntsa[t][s][a] * at_rew[t][s][a]
            rg_tab = np.vstack((rg_tab, np.array([[rew_count]])))

            # --- Update node counter
            node_len += 1
            x_id += 1
            x_node_id = x_id

            # --- create nbr list
            x_xp = np.zeros((1, NB_DIM))
            x_xp.fill(-1)
            xxp_tab = np.vstack((xxp_tab, x_xp))

        # ------- node x'
        if t+1 < HORIZON:
            t3 = np.where(ntsa[t+1][sp] > 0)
            sp_ap = np.vstack(t3)[0]
            for ii in range(sp_ap.shape[0]):
                ap = sp_ap[ii]

                # ---- obs
                op = np.zeros(o_dim)
                o_nbr_indexes = o_nbr[sp][o_nbr[sp] > -1]
                o_nbr_len = o_nbr_indexes.shape[0]
                o1 = obs[t+1][o_nbr_indexes]
                op[0:o_nbr_len] = o1
                tot_len = s_dim + o_dim + a_dim #op.shape[0]
                sp_op_ap = np.zeros((1, tot_len))
                sp_op_ap[0][0:s_dim] = sp
                sp_op_ap[0][s_dim:s_dim+o_dim] = op
                sp_op_ap[0][s_dim+o_dim:tot_len] = ap

                # ---- find node in graph
                sim_indx = hash_map_act(x_tab, sp_op_ap)
                if sim_indx != x_node_id:
                    if sim_indx != -1:
                        # ------ Node found in graph
                        # --- update t
                        sim_t_tab = t_tab[sim_indx][0]
                        if sim_t_tab < t+1:
                            t_tab[sim_indx][0] = t+1

                        # ---- Update values
                        idv_val = val_fn[t+1][sp][ap]
                        fg_val[sim_indx] = max(fg_val[sim_indx][0], idv_val)
                        fg_soa[sim_indx] = np.maximum(fg_soa[sim_indx], val_fn[t+1][sp])


                        # --- Atomic reward multiplied with count
                        rew_count = ntsa[t + 1][sp][ap] * at_rew[t + 1][sp][ap]
                        rg_tab[sim_indx] = max(rg_tab[sim_indx][0], rew_count)

                        # ---- Update nbr list
                        if sim_indx not in xxp_tab[x_node_id] and x_node_id != sim_indx:
                            ind1_list = np.where(xxp_tab[x_node_id] == -1)[0]
                            if ind1_list.shape[0] > 0:
                                ind1 = ind1_list.min()
                                xxp_tab[x_node_id][ind1] = sim_indx
                    else:
                        # ------ Node not found in graph
                        # --- Create
                        x_tab = np.vstack((x_tab, sp_op_ap))
                        t_tab = np.vstack((t_tab, np.array([[t + 1]])))

                        # --- Update values
                        idv_val = val_fn[t + 1][sp][ap]
                        fg_val = np.vstack((fg_val, np.array([[idv_val]])))
                        fg_soa = np.vstack((fg_soa, val_fn[t+1][sp].reshape(1, num_action)))

                        # --- Atomic reward multiplied with count
                        rew_count = ntsa[t + 1][sp][ap] * at_rew[t + 1][sp][ap]
                        rg_tab = np.vstack((rg_tab, np.array([[rew_count]])))

                        # ---- Update node counter
                        node_len += 1
                        x_id += 1

                        # --- create nbr list
                        x_xp = np.zeros((1, NB_DIM))
                        x_xp.fill(-1)
                        xxp_tab = np.vstack((xxp_tab, x_xp))

                        # ---- Update nbr list for soa
                        if x_id not in xxp_tab[x_node_id]:
                            ind1_list = np.where(xxp_tab[x_node_id] == -1)[0]
                            if ind1_list.shape[0] > 0:
                                ind1 = ind1_list.min()
                                xxp_tab[x_node_id][ind1] = x_id


    # ------- sort time index
    t_sorted = np.zeros(t_tab.shape[0])
    t_sorted.fill(-1)
    counter = 0
    for t in range(HORIZON-1, -1, -1):
        t1 = np.where(t_tab == t)[0]
        if t1.shape[0] > 0:
            t_sorted[counter:counter+t1.shape[0]] = t1
            counter += t1.shape[0]

    return x_tab, t_tab, xxp_tab, rg_tab, fg_val, t_sorted, fg_soa

@nb.njit(fastmath=True, cache=True, nogil=True)
def update_val_fn(s_dim, o_dim, ntsa, buff_x, val_fn):

    num_action = ntsa.shape[2]
    train_val = np.empty((0, num_action))
    train_nsa = np.empty((0, num_action))
    train_x = np.empty((0, s_dim+o_dim))
    x_tmp1 = np.zeros((1, s_dim + o_dim))

    for i in range(buff_x.shape[0]):
        x = buff_x[i]
        t = int(x[0])
        s = int(x[1])
        x_tmp1[0][0] = s
        x_tmp1[0][1:] = x[2:]
        # -- idv
        train_val = np.vstack((train_val, val_fn[t][s].reshape(1, num_action)))
        train_x = np.vstack((train_x, x_tmp1))
        train_nsa = np.vstack((train_nsa, ntsa[t][s].reshape(1, num_action)))

    return train_x, train_val, train_nsa

@nb.njit(fastmath=True, cache=True, nogil=True)
def prep_train_data(s_dim, o_dim, ntsa, buff_x, val_fn):

    num_action = ntsa.shape[2]
    fg_index = np.zeros(buff_x.shape[0])
    fg_index.fill(-1)

    train_val = np.empty((0, num_action))
    train_nsa = np.empty((0, num_action))
    train_x = np.empty((0, s_dim + o_dim))
    x_tmp1 = np.zeros((1, s_dim + o_dim))

    for i in range(buff_x.shape[0]):
        x = buff_x[i]
        t = int(x[0])
        s = int(x[1])
        x_tmp1[0][0] = s
        o = x[2:]
        x_tmp1[0][1:] = o

        # --- Prep
        train_val = np.vstack((train_val, val_fn[t][s].reshape(1, num_action)))
        train_x = np.vstack((train_x, x_tmp1))
        train_nsa = np.vstack((train_nsa, ntsa[t][s].reshape(1, num_action)))

    return train_x, train_val, train_nsa, fg_index

@nb.njit(fastmath=True, cache=True, nogil=True)
def prep_train_data_dec(s_dim, o_dim, ntsa, buff_x, val_fn):

    fg_index = np.zeros(buff_x.shape[0])
    fg_index.fill(-1)

    train_val = val_fn
    train_nsa = ntsa
    train_x = buff_x

    return train_x, train_val, train_nsa, fg_index

@nb.njit(fastmath=True, cache=True, nogil=True)
def discounted_return(reward_list):
    return_so_far = 0.0
    tot_time = reward_list.shape[0]
    tmpReturn = np.zeros(tot_time)
    k = 0
    for t in range(tot_time - 1, -1, -1):
        return_so_far = reward_list[t] + DISCOUNT * return_so_far
        tmpReturn[k] = return_so_far
        k += 1
    tmpReturn = np.flip(tmpReturn)
    return tmpReturn

@nb.njit(fastmath=True, cache=True, nogil=True)
def nb_mean(arr):

    # row = arr.shape[0]
    col = arr.shape[1]
    mean = np.zeros(col)
    for a in range(col):
        t1 = arr[:, a]
        t1 = t1.sum()/NEAREST_NBR
        mean[a] = t1
    return mean
    #set_trace()

@nb.njit(fastmath=True, cache=True, nogil=True)
def add_miss_values(fg_index, buff_x, x_tab_new, fg_soa, train_fg, a_dim):

    for i in range(fg_index.shape[0]):
        if fg_index[i] == -1:
            x = buff_x[i][1:]
            x = x.reshape(1, x.shape[0])
            x_arr = x_tab_new[:, 0:-a_dim]
            index, nbr_index = hash_map_nbr(x_arr, x)
            if index != -1:
                val = fg_soa[index]
                fg_index[i] = index
            else:
                l1 = fg_soa[nbr_index]
                val = nb_mean(l1)
            train_fg[i] = val

    return train_fg, fg_index

@nb.njit(fastmath=True, cache=True, nogil=True)
def create_train_fg(buff_x, x_tab_new, fg_soa, s_dim, o_dim, a_dim, num_states, num_actions, o_nbr_index, fg_val, train_fg):

    train_fg = np.zeros((buff_x.shape[0], num_states, num_actions))
    buff_x_size = buff_x.shape[0]
    for t in range(buff_x_size):
        x = buff_x[t]
        for s in range(num_states):
            o = np.zeros(o_dim)
            o_indx = o_nbr_index[s]
            o_len = o_indx.shape[0]
            o1 = x[o_indx]
            o[0:o_len] = o1
            for a in range(num_actions):
                soa = np.zeros((1, 1 + o_dim + 1))
                soa[0][0] = s
                soa[0][1:1+o_dim] = o
                soa[0][1+o_dim:] = a
                index, nbr_index = hash_map_act_nbr(x_tab_new, soa)
                if index != -1:
                    val = fg_val[index][0]
                else:
                    l1 = fg_val[nbr_index]
                    val = nb_mean(l1)[0]
                train_fg[t][s][a] = val

    return train_fg

@nb.njit(fastmath=True, cache=True, nogil=True)
def create_train_fg_all_actions(buff_x, x_tab_new, fg_soa, s_dim, o_dim, a_dim, num_states, num_actions, o_nbr_index, fg_val, train_fg):

    buff_x_size = buff_x.shape[0]
    for t in range(buff_x_size):
        x = buff_x[t]
        for s in range(num_states):
            o = np.zeros(o_dim)
            o_indx = o_nbr_index[s]
            o_len = o_indx.shape[0]
            o1 = x[o_indx]
            o[0:o_len] = o1
            for a in range(num_actions):
                soa = np.zeros((1, 1 + o_dim + 1))
                soa[0][0] = s
                soa[0][1:1+o_dim] = o
                soa[0][1+o_dim:] = a
                index, nbr_index = hash_map_act_nbr(x_tab_new, soa)
                if index != -1:
                    val = fg_soa[index]
                else:
                    l1 = fg_soa[nbr_index]
                    val = nb_mean(l1)
                train_fg[t][s] = val

    return train_fg

@nb.njit(fastmath=True, cache=True, nogil=True)
def compute_idv(buff_at_rw, buff_ntsa, epoch, num_states, num_actions, meanAds, stdAds):

    horizon = buff_at_rw.shape[0]
    # ---------- Compute Q & V - Values ----------- #
    # ------- Q ------ #
    q_tmp = np.zeros((horizon, num_states, num_actions))
    v_tmp = np.zeros((horizon+1, num_states))

    for t in range(horizon -1, -1, -1):
        for s in range(num_states):
            for a in range(num_actions):
                for tp in range(t+1, horizon+1):
                    for sp in range(num_states):
                        q_tmp[t][s][a] = buff_at_rw[t][s][a] + DISCOUNT * v_tmp[tp][sp]

            # -------- V
            nb_max = np.maximum(np.sum(buff_ntsa[t][s]), TINY)
            v_tmp[t][s] = np.sum( q_tmp[t][s][:] * buff_ntsa[t][s][:]) / nb_max


    # if epoch == 1:
    #     meanAds = np.sum(q_tmp * buff_ntsa) / horizon
    #     stdAds = np.sqrt(np.sum(np.square(q_tmp - meanAds) * buff_ntsa) / horizon)
    # else:
    #     meanAds1 = np.sum(q_tmp * buff_ntsa) / horizon
    #     stdAds1 = np.sqrt(np.sum(np.square(q_tmp - meanAds) * buff_ntsa) / horizon)
    #
    #     meanAds = 0.9 * meanAds1 + 0.1 * meanAds
    #     stdAds = 0.9 * stdAds1 + 0.1 * stdAds
    #
    # q_tmp = (q_tmp - meanAds)/(stdAds + TINY)
    # v_tmp = (v_tmp - meanAds)/(stdAds + TINY)
    #
    # # ------------ Advantage ------------ #
    # Adv = np.zeros((horizon, num_states, num_actions))
    # for t in range(horizon):
    #     for s in range(num_states):
    #         for a in range(num_actions):
    #             Adv[t][s][a] = q_tmp[t][s][a]
    #             Adv[t][s][a] -= v_tmp[t][s]
    # return Adv, meanAds, stdAds
    return q_tmp, -1 , -1


class log:

    def __init__(self, fl):
        self.opfile = fl
        if os.path.exists(self.opfile):
            os.remove(self.opfile)
        # f = open(self.opfile, 'w')
        # f.write("test")
        # f.close()

    def writeln(self, msg):
        file = self.opfile
        print(str(msg))
        with open(file, "a") as f:
            f.write("\n"+str(msg))

    def write(self, msg):
        file = self.opfile
        print(str(msg),)
        with open(file, "a") as f:
            f.write(str(msg))

def main():
    pass


# =============================================================================== #

if __name__ == '__main__':
    main()
