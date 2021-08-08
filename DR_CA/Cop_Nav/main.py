"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James ArambamBill
Date   : 18 Jul 2020
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys
from ipdb import set_trace
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
from multiagent.environment import MultiAgentEnv
from multiagent.agent.random import random_policy
from multiagent.agent.diff_mid import diff_mid
from multiagent.agent.diff_max import diff_max
from multiagent.agent.mean_field import mean_field
from multiagent.agent.global_count import global_count
from multiagent.agent.appx_dr_colby import appx_dr_colby
import multiagent.scenarios as scenarios
from parameters import EPISODES, HORIZON, AGENT, SEED, LOAD_MODEL, GRID, NUM_ACTIONS, BATCH_SIZE, SAVE_MODEL, SHOULD_LOG, NUM_CORES
from utils import log, deleteDir
import torch
import numpy as np
np.random.seed(SEED)
torch.set_num_threads(NUM_CORES)

# =============================== Variables ================================== #


# ============================================================================ #

def init(pro_folder, dir_name):

    os.system("mkdir "+pro_folder+"/log/")
    os.system("mkdir "+pro_folder+"/log/" + dir_name)
    if LOAD_MODEL is False:
        deleteDir(pro_folder+"/log/"+dir_name + "/plots/")
    os.system("cp "+pro_folder+"/"+"parameters.py "+pro_folder+"/log/"+dir_name+"/")
    os.system("mkdir "+pro_folder+"/log/"+dir_name+"/plots")
    os.system("mkdir "+pro_folder+"/log/" + dir_name + "/model")

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_spread.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)

    # -------------------------- #
    pro_folder = os.getcwd()
    dir_name = AGENT
    init(pro_folder, dir_name)
    if LOAD_MODEL:
        lg = log(pro_folder + "/log/" + dir_name + "/log_inf"+".txt")
    else:
        lg = log(pro_folder + "/log/"+dir_name+"/log"+".txt")
    agent = None
    if AGENT == "random":
        # policies = [random_policy(env, i) for i in range(env.n)]
        agent = random_policy(dim_c=env.world.dim_c, lg=lg, dir_name=dir_name, pro_folder=pro_folder)
    elif AGENT == "diff_mid":
        agent = diff_mid(dim_c=env.world.dim_c, lg=lg, dir_name=dir_name, pro_folder=pro_folder)
    elif AGENT == "diff_max":
        agent = diff_max(dim_c=env.world.dim_c, lg=lg, dir_name=dir_name, pro_folder=pro_folder)
    elif AGENT == "mean_field":
        agent = mean_field(dim_c=env.world.dim_c, lg=lg, dir_name=dir_name, pro_folder=pro_folder)
    elif AGENT == "global_count":
        agent = global_count(dim_c=env.world.dim_c, lg=lg, dir_name=dir_name, pro_folder=pro_folder)
    elif AGENT == "appx_dr_colby":
        agent = appx_dr_colby(dim_c=env.world.dim_c, lg=lg, dir_name=dir_name, pro_folder=pro_folder)
    else:
        print("Error: Agent not implemented")
        exit()

    episode_rewards = []  # sum of rewards for all agents
    episode_dist = []  # sum of rewards for all agents
    episode_col = []  # sum of rewards for all agents


    for e in range(1, EPISODES+1):
        # ------- Buffer
        buff_nts = np.zeros((HORIZON, GRID*GRID, GRID*GRID), dtype=int)
        buff_ntsa = np.zeros((HORIZON, GRID*GRID, GRID*GRID, NUM_ACTIONS), dtype=int)
        buff_rt = np.zeros(HORIZON)
        buff_act_prob = np.zeros((HORIZON, NUM_ACTIONS))
        obs_n = env.reset()
        nts = env.init_count()
        rt_sum = 0
        dist_sum = 0
        col_sum = 0

        episode_avg_dist = []
        episode_avg_col = []
        episode_avg_rw = []

        for t in range(0, HORIZON):
            # lg.writeln("# ---------------------------------- #")
            act_n, act_id, action_prob = agent.action(obs_n, nts)
            obs_n, rt, done_n, _, pos_tile_n, lm_pos_tile, ntsa, tot_dist, num_col, reward_n, avg_dist, avg_col, avg_rw  = env.step(act_n, act_id)

            if "mean_field" in AGENT:
                agent.get_action_p(obs_n)

            elif "appx_dr_colby" in AGENT:

                agent.update_buffer(t, nts, act_id, rt)


            rt_sum += rt
            dist_sum += tot_dist
            col_sum += num_col

            episode_avg_dist.append(avg_dist)
            episode_avg_col.append(avg_col)
            episode_avg_rw.append(avg_rw)

            # ---- Store Buffer
            nts_new = ntsa.sum(2).copy()
            buff_nts[t] = nts_new.copy()
            buff_ntsa[t] = ntsa.copy()
            buff_rt[t] = rt
            rt_sum += rt
            buff_act_prob[t] = action_prob
            nts = nts_new.copy()

        episode_rewards.append(rt_sum)
        episode_dist.append(dist_sum)
        episode_col.append(col_sum)

        agent.store_rollouts(buff_nts=buff_nts, buff_ntsa=buff_ntsa, buff_rt=buff_rt)

        if e % BATCH_SIZE == 0:
            if LOAD_MODEL == False:
                agent.train(ep=e)
                lg.writeln("   Loss : " + str(agent.loss))
            agent.clear_buffer()

        # ------ Save model
        # if e % SAVE_MODEL == 0 and LOAD_MODEL == False:
        #     agent.save_model(rt_sum, e)

        # Log
        tmp = buff_ntsa.sum(1).sum(1)
        mean_act_count = tmp[:, :].mean(0)
        if e % SHOULD_LOG == 0:

            # mean_rw = np.mean(episode_rewards)
            # mean_dist = np.mean(episode_dist)
            # mean_col = np.mean(episode_col)

            av_rt = np.mean(episode_avg_rw)
            agent.log(e, av_rt, buff_act_prob, mean_act_count)

            lg.writeln("\n# ---------------------------------- #")
            lg.writeln("Episode : " + str(e))
            lg.writeln("   Avg. Reward : " + str(round(np.mean(episode_avg_rw), 3)))
            lg.writeln("   Avg. Distance : " + str(round(np.mean(episode_avg_dist), 3)))
            lg.writeln("   Avg. Collisions : " + str(round(np.mean(episode_avg_col), 3)))

            lg.writeln("   Total Reward : " + str(round(rt_sum, 3)))
            lg.writeln("   Total Distance : " + str(round(dist_sum, 3)))
            lg.writeln("   Total Collisons : " + str(round(col_sum, 3)))

            lg.writeln("   Avg. Action count : ")
            lg.writeln("   " + str(mean_act_count))

            episode_rewards = []  # sum of rewards for all agents
            episode_dist = []  # sum of rewards for all agents
            episode_col = []  # sum of rewards for all agents








# ============================================================================ #

if __name__ == '__main__':

    main()