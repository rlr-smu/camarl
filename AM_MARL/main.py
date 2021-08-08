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
from parameters import RENDER, HORIZON, EPISODES, SEED, LOAD_MODEL, AGENT_NAME, GRID, SHOULD_LOG, LEARNING_RATE, ENTROPY_WEIGHT, SAVE_MODEL, ENV_NAME, NB_DIM, MAX_ITR_VP, BATCH_SIZE, DISCOUNT
from utils import deleteDir, log, logRunTime, get_one_hot, prep_train_data
import argparse
from environment import environment
from agents.random import random_agent
from agents.vpg import vpg
from agents.idv import idv
from agents.idv_dec import idv_dec
import matplotlib.pyplot as plt

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
    if agent_name == "vpg":
        agent = vpg(config=config)
    elif agent_name == "idv":
        agent = idv(config=config)
    elif agent_name == "idv_dec":
        agent = idv_dec(config=config)
    elif agent_name == "random":
        agent = random_agent(config=config)
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

    # ----- RunTime
    tm = logRunTime(init_time=time.time())
    runTime = 0

    # ----- Memory
    x_mem = np.empty((0, s_dim + o_dim), dtype=NP_DTYPE_D)
    v_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)
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

        train_x, train_val, train_nsa, _ = prep_train_data(s_dim, o_dim, buff_ntsa, buff_x, val_fn)
        lg.writeln(" Prep TrainData Time: " + tm.logTime())

        # ---------- Add in memory
        x_mem = np.vstack((x_mem, train_x))
        v_mem = np.vstack((v_mem, train_val))
        n_mem = np.vstack((n_mem, train_nsa))

        # ----------- Training
        if LOAD_MODEL is False and x_mem.shape[0] > BATCH_SIZE:
            agent.train(x_mem = x_mem, v_mem = v_mem, n_mem = n_mem)

            # ----- Clear Memory
            x_mem = np.empty((0, s_dim + o_dim), dtype=NP_DTYPE_D)
            v_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)
            n_mem = np.empty((0, num_actions), dtype=NP_DTYPE_D)

            lg.writeln(" Train time: " + tm.logTime())
            if ep % SAVE_MODEL == 0:
                agent.save_model(ep, rt_sum)

        # -------- Log
        if ep % SHOULD_LOG == 0:
            agent.log(ep, rt_sum)
            agent.log_agent(ep)
            lg.writeln("\n Return: " + str(rt_sum))

        ep_end = tm.now()
        ep_run = tm.getRuntime(ep_start, ep_end)
        lg.writeln("\n Episode Runtime: "+str(round(ep_run, 2))+" sec")
        runTime += ep_run

        lg.writeln("\n\n Total Runtime: " + str(round(runTime, 2))+" sec")


# =============================================================================== #

if __name__ == '__main__':
    main()
