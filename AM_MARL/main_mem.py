"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 26 May 2021
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
import numpy as np
from parameters import RENDER, HORIZON, EPISODES, SEED, LOAD_MODEL, AGENT_NAME, GRID, SHOULD_LOG, LEARNING_RATE, ENTROPY_WEIGHT, SAVE_MODEL, ENV_NAME, NB_DIM, MAX_ITR_VP, BATCH_SIZE, DISCOUNT, LAMBDA
from utils import deleteDir, log, update_mem_graph,  prep_train_data,logRunTime, get_one_hot, add_miss_values
import argparse
from environment import environment
from agents.vpg_mem import vpg_mem
from agents.idv_mem import idv_mem
from cy_utils import value_propagation


# =============================== Variables ================================== #
np.random.seed(SEED)
parser = argparse.ArgumentParser(description=None)
parser.add_argument('-a', '--agent_name', default="-1", type=str)
args = parser.parse_args()


# ================================ Global Variables ================================= #
NP_DTYPE_I = np.int
NP_DTYPE_D = np.float64

# ============================================================================ #

def init_agent(agent_name, config):

    agent = None
    if agent_name == "vpg_mem":
        agent = vpg_mem(config=config)
    elif agent_name == "idv_mem":
        agent = idv_mem(config=config)
    else:
        print("Agent not found !")
        exit()
    return agent

def init(pro_folder, dir_name):

    os.system("mkdir "+pro_folder+"/log/")
    os.system("mkdir "+pro_folder+"/log/" + dir_name)
    if LOAD_MODEL is False:
        deleteDir(pro_folder+"/log/"+dir_name + "/plots/")
    os.system("cp "+pro_folder+"/"+"parameters.py "+pro_folder+"/log/"+dir_name+"/")
    os.system("mkdir "+pro_folder+"/log/"+dir_name+"/plots")
    os.system("mkdir "+pro_folder+"/log/" + dir_name + "/model")

def main():


    # ----- Agent_Name
    if args.agent_name == "-1":
        agent_name = AGENT_NAME
    else:
        agent_name = args.agent_name

    # --------- Init
    pro_folder = os.getcwd()
    dir_name_hash = {}
    dir_name_hash.update({"lr": str(LEARNING_RATE)})
    dir_name_hash.update({"ld": str(LAMBDA)})
    tstr = ""
    for k in dir_name_hash:
        tstr += "_"+k+"_"+dir_name_hash[k]
    dir_name = agent_name + tstr
    init(pro_folder, dir_name)
    if LOAD_MODEL:
        lg = log(pro_folder + "/log/" + dir_name + "/log_inf"+".txt")
    else:
        lg = log(pro_folder + "/log/"+dir_name+"/log"+".txt")

    # -------- Environment
    env = environment(env_name=ENV_NAME)
    num_actions = env.num_actions
    nb_agents = env.total_agents
    num_states = env.num_states
    s_dim = env.state_dim
    a_dim = env.action_dim
    # at_rew = env.atomic_reward
    o_dim = env.obs_dim
    o_nbr = env.obs_nbr.astype(NP_DTYPE_I)
    one_hot_hash = get_one_hot(num_states)

    # --------- Agents
    config = {}
    config['num_states'] = num_states
    config['num_actions'] = num_actions
    config['num_agents'] = nb_agents
    config['pro_folder'] = pro_folder
    config['dir_name'] = dir_name
    config['agent_name'] = agent_name
    config['lg'] = lg
    config['s_dim'] = s_dim
    config['a_dim'] = a_dim
    config['o_dim'] = o_dim
    config['o_nbr'] = o_nbr
    config['one_hot_hash'] = one_hot_hash

    agent = init_agent(agent_name, config)

    # --------- Memory Graph
    x_tab = np.empty((0, s_dim+o_dim+a_dim), dtype=NP_DTYPE_D)
    t_tab = np.empty((0, 1), dtype=NP_DTYPE_D)
    xxp_tab = np.empty((0, NB_DIM), dtype=NP_DTYPE_D)
    rg_tab = np.zeros((0, num_actions), dtype=NP_DTYPE_D)
    fg_val = np.empty((0, num_actions), dtype=NP_DTYPE_D)
    fg_soa = np.empty((0, num_actions), dtype=NP_DTYPE_D)
    graph_hit = 0

    # ----- RunTime
    tm = logRunTime(init_time=time.time())
    runTime = 0

    # ----- Memory
    x_mem = np.empty((0, s_dim + o_dim), dtype=NP_DTYPE_D)
    v_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)
    g_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)
    n_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)

    # -------- Start Simulation
    for ep in range(1, EPISODES + 1):
        ep_start = tm.now()
        lg.writeln("\n# --------------------- #")
        lg.writeln("Episode: "+str(ep))
        rt_sum = 0

        nts = env.reset()

        # ------ Buffers
        buff_rt = np.empty((0), dtype=NP_DTYPE_D)
        buff_at_rw = np.empty((0, num_states, num_actions), dtype=NP_DTYPE_D)
        buff_nts = np.empty((0, num_states), dtype=NP_DTYPE_I)
        buff_ntsa = np.empty((0, num_states, num_actions), dtype=NP_DTYPE_I)
        buff_ntsas = np.empty((0, num_states, num_actions, num_states), dtype=NP_DTYPE_I)
        buff_x = np.empty((0, 1 + s_dim  + o_dim))

        tm.logTime()
        for t in range(HORIZON):
            if RENDER:
                env.render(mode='human', highlight=False)
                time.sleep(0.1)

            buff_x, ac = agent.get_action(t, nts, buff_x)
            nts_new, ntsa, ntsas, action, rew, at_reward, done, _ = env.step(ac)
            lg.writeln(str(t)+" "+str(nts)+" "+str(rew))

            rt_sum += rew
            buff_rt = np.hstack((buff_rt, rew))
            buff_at_rw = np.vstack((buff_at_rw, at_reward))
            buff_nts = np.vstack((buff_nts, nts))
            buff_ntsa = np.vstack((buff_ntsa, np.expand_dims(ntsa, axis=0)))
            buff_ntsas = np.vstack((buff_ntsas, np.expand_dims(ntsas, axis=0)))
            nts = nts_new.copy()
            if done:
                break

        lg.writeln("\n Simulation Time: " + tm.logTime())

        # -------- Compute Value Function
        val_fn = agent.compute_val_fn(buff_rt=buff_rt, buff_at_rw=buff_at_rw, buff_ntsa=buff_ntsa, buff_ntsas=buff_ntsas)
        lg.writeln(" Val Fn Time: " + tm.logTime())

        train_x, train_val, train_nsa, fg_index = prep_train_data(s_dim, o_dim, buff_ntsa, buff_x, val_fn)
        lg.writeln(" Prep TrainData Time: " + tm.logTime())


        # -------- Update Memory Graph
        x_tab_new, t_tab_new, xxp_tab_new, rg_tab, fg_val, sorted_time, fg_index, fg_soa = update_mem_graph(buff_at_rw, buff_x, buff_ntsas, buff_ntsa, buff_nts, s_dim, a_dim, o_dim, x_tab, t_tab, xxp_tab, rg_tab, fg_val, o_nbr, fg_index, fg_soa, val_fn)

        # set_trace()

        lg.writeln(" Graph time: " + tm.logTime())

        x_tab = x_tab_new
        t_tab = t_tab_new
        xxp_tab = xxp_tab_new

        node_len = x_tab.shape[0]
        # -- Avg nbr
        t1 = np.where((xxp_tab > -1), 1, 0)
        t2 = t1.sum(1)/xxp_tab.shape[1]
        avg_nbr = t2.mean()

        # ----------- Value Propagation
        conv_itr = np.zeros(MAX_ITR_VP)
        fg_soa, train_fg, conv_itr = value_propagation(t_tab_new, x_tab_new, xxp_tab_new, fg_soa, rg_tab, sorted_time, MAX_ITR_VP, fg_index, DISCOUNT, conv_itr)
        lg.writeln(" Value prop time: " + tm.logTime())

        # ---------- Add Values for miss nodes
        train_fg, fg_index = add_miss_values(fg_index, buff_x, x_tab_new, fg_soa, train_fg, a_dim)
        graph_hit = float(fg_index[fg_index != -1].shape[0]) / fg_index.shape[0]
        lg.writeln(" Add miss value time: " + tm.logTime())

        # ---------- Add in memory
        x_mem = np.vstack((x_mem, train_x))
        v_mem = np.vstack((v_mem, train_val))
        n_mem = np.vstack((n_mem, train_nsa))
        g_mem = np.vstack((g_mem, train_fg))

        # ----------- Training
        if LOAD_MODEL is False and x_mem.shape[0] > BATCH_SIZE:
            agent.train(x_mem = x_mem, v_mem = v_mem, n_mem = n_mem, g_mem = g_mem)

            # ----- Clear Memory
            x_mem = np.empty((0, s_dim + o_dim), dtype=NP_DTYPE_D)
            v_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)
            g_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)
            n_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)

            lg.writeln(" Train time: " + tm.logTime())
            if ep % SAVE_MODEL == 0:
                agent.save_model(ep, rt_sum)

        # -------- Log
        if ep % SHOULD_LOG == 0:
            agent.log(ep, rt_sum)
            agent.log_agent(ep, node_len, avg_nbr, graph_hit)
            lg.writeln("\n Return: " + str(rt_sum))
            lg.writeln(" Node size: " + str(node_len))
            lg.writeln(" Avg nbr: " + str(round(avg_nbr, 2)))
            lg.writeln(" Graph hit: " + str(round(graph_hit, 2)))

        ep_end = tm.now()
        ep_run = tm.getRuntime(ep_start, ep_end)
        lg.writeln("\n Episode Runtime: "+str(round(ep_run, 2))+" sec")
        runTime += ep_run

        lg.writeln("\n\n Total Runtime: " + str(round(runTime, 2))+" sec")


# =============================================================================== #

if __name__ == '__main__':
    main()
