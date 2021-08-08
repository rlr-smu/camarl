"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 03 Jun 2021
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
from tensorboardX import SummaryWriter
import torch as tc
import numpy as np
from parameters import LARGE, TOTAL_AGENTS
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

class baseAgent(object):

    def __init__(self,  config=None):

        self.num_states = config['num_states']
        self.num_actions = config['num_actions']
        self.num_agents = config['num_agents']
        self.agent_name = config['agent_name']
        self.pro_folder = config['pro_folder']
        self.dir_name = config['dir_name']
        self.lg = config['lg']
        self.s_dim = config['s_dim']
        self.a_dim = config['a_dim']
        self.o_dim =  config['o_dim']
        self.o_nbr = config['o_nbr']
        self.one_hot_hash = config['one_hot_hash']
        self.one_hot_dim = self.one_hot_hash.shape[0]
        self.action_space = [ _ for _ in range(self.num_actions)]
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.max_return = -1 * LARGE
        self.entropy = -1

        self.actor = None

    def get_action(self, t, nts, buff_x):

        prob_all = np.empty((0, self.num_actions))
        for s in range(self.num_states):
            s_m = s
            s_1hot = self.one_hot_hash[s_m]
            o = np.zeros(self.o_dim)
            o_nbr_indexes = self.o_nbr[s_m][self.o_nbr[s_m] > -1]
            o_nbr_len = o_nbr_indexes.shape[0]
            o1 = nts[o_nbr_indexes]
            o[0:o_nbr_len] = o1

            # --- normalize obs count
            o = o/TOTAL_AGENTS
            x_1hot = np.hstack((s_1hot, o))
            x_1hot = x_1hot.reshape(1, x_1hot.shape[0])
            pi_logit, _ = self.actor(x_1hot)
            prob = F.softmax(pi_logit).data.numpy()
            prob_all = np.vstack((prob_all, prob))
            x = np.hstack((t, s_m, o))
            buff_x = np.vstack((buff_x, x))

        return buff_x, prob_all

    def log(self, ep, rt_sum):

        self.writer.add_scalar('Return/', rt_sum, ep)
        if self.entropy != -1:
            self.writer.add_scalar('Avg_Entropy/', self.entropy, ep)

    def compute_val_fn(self, buff_rt=None, buff_at_rw=None, buff_ntsa=None, buff_ntsas=None):

        self.total_time = buff_rt.shape[0]
        val_fn = np.zeros((self.total_time, self.num_states, self.num_actions))
        return val_fn

    def save_model(self, ep, rt_sum):

        if rt_sum >= self.max_return:
            self.max_return = rt_sum
            tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor' + "_" + self.agent_name + ".pt")

    def train(self, x_mem = None, v_mem = None, n_mem = None, g_mem = None):

        pass

class baseAgent_dec(object):

    def __init__(self,  config=None):

        self.num_states = config['num_states']
        self.num_actions = config['num_actions']
        self.num_agents = config['num_agents']
        self.agent_name = config['agent_name']
        self.pro_folder = config['pro_folder']
        self.dir_name = config['dir_name']
        self.lg = config['lg']
        self.s_dim = config['s_dim']
        self.a_dim = config['a_dim']
        self.o_dim =  config['o_dim']
        self.o_nbr = config['o_nbr']
        self.one_hot_hash = config['one_hot_hash']
        self.one_hot_dim = self.one_hot_hash.shape[0]
        self.action_space = [ _ for _ in range(self.num_actions)]
        self.writer = SummaryWriter(self.pro_folder + "/log/" + self.dir_name + "/plots/")
        self.max_return = -1 * LARGE
        self.entropy = -1

        self.actor = None

        self.o_nbr_index = []
        for s in range(self.num_states):
            s_m = s
            indexes = self.o_nbr[s_m][self.o_nbr[s_m] > -1]
            self.o_nbr_index.append(indexes)
        self.o_nbr_index = np.array(self.o_nbr_index)

        # --- MemGraph
        os.system("mkdir " + self.pro_folder + "/log/" + self.dir_name + "/memGraph")
        self.G = nx.DiGraph()
        self.gCounter = 1
        self.minusOne = 0

    def get_action(self, t, nts, buff_x):

        # x = np.random.randint(0, 10, size=(3, self.num_states))

        x = nts.reshape(1, nts.shape[0])
        x = x / TOTAL_AGENTS
        buff_x = np.vstack((buff_x, x))


        logit, _ = self.actor(x)
        prob_all = F.softmax(logit[0]).data.numpy()
        return buff_x, prob_all

    def log(self, ep, rt_sum):

        self.writer.add_scalar('Return/', rt_sum, ep)
        if self.entropy != -1:
            self.writer.add_scalar('Avg_Entropy/', self.entropy, ep)

    def compute_val_fn(self, buff_rt=None, buff_at_rw=None, buff_ntsa=None, buff_ntsas=None):

        self.total_time = buff_rt.shape[0]
        val_fn = np.zeros((self.total_time, self.num_states, self.num_actions))
        return val_fn

    def save_model(self, ep, rt_sum):

        if rt_sum >= self.max_return:
            self.max_return = rt_sum
            tc.save(self.actor, self.pro_folder + '/log/' + self.dir_name + '/model/model_actor_dec' + "_" + self.agent_name + ".pt")

    def train(self, x_mem = None, v_mem = None, n_mem = None, g_mem = None):

        pass

    def plot_memGraph(self, xxp_tab):

        t1 = xxp_tab[xxp_tab == -1]
        minusOne_new = t1.shape[0]

        if minusOne_new != self.minusOne:
            plt.clf()
            for i in range(xxp_tab.shape[0]):
                n1 = i
                for j in range(xxp_tab.shape[1]):
                    n2 = xxp_tab[i][j]
                    if n2 != -1:
                        self.G.add_edge(int(n1), int(n2))
            nx.draw_networkx(self.G, with_labels=True)
            plt.savefig(self.pro_folder+"/"+"log/"+self.dir_name+"/memGraph/"+str(self.gCounter)+".png")
            self.gCounter += 1
            self.minusOne = minusOne_new
            set_trace()


def main():
    print("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
