""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, scr, tools
from bluesky.tools.areafilter import checkInside, areas

from parameters import EPISODES, HORIZON, VERBOSE, SEED, EP_BATCH_SIZE, SHOULD_LOG, SAVE_MODEL, LOAD_MODEL, AGENT, MAP_ID, TEST_ID, UPDATE_INTV, SAC_UPDATE_AFTER, SAC_UPDATE_EVERY, SAC_BATCH_SIZE, SAC_WARMUP, SAC_TRAIN_EPOCH, LEARNING_RATE, SAC_REPLAY_BUFFER, SAC_TEST_EVERY, SAC_EVAL_EP, SAC_TRAIN_EP
from count_env_cont import countAirSim


from ipdb import set_trace
from data import syn_data


from agents.random_agent import random_agent, min_agent, max_agent
# --- sac
# import argparse
import torch
import agents.sac.core as core
from agents.sac.sac import sac_dr_global, sac_dr_indv, sac_rw_indv, sac_random
from utils import display, log, deleteDir, real_time
import numpy as np
import os
from auxLib3 import now, getRuntime
import torch as tc
import time
# plt.ion()


def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name':     'countSim_sac_inf',
        'plugin_type':     'sim',
        'update_interval': UPDATE_INTV,
        'update':          update
        #'preupdate':       preupdate,
        #'reset':         reset
        }
    stackfunctions = {     }

    # ------- RL Environment
    global cenv, nt, flag, ep, t, start, create_flag, tot_reward, dt, agent
    global buff_nt, buff_ntellv, buff_rt, dir_name, lg, buff_act_prob
    global buff_nm_conf, pro_folder, plot_conf, plot_goal_reached, plot_conf_mean, mean_action
    global start_time, curr_runtime#, ax

    global sampling_time, rollout_time, training_time, mod1, mod2, mod3, reset_st_time, tot_runtime_2, else_count, tot_runtime_list, reset_en_time

    # --- sac
    global env_intr, EvalEpCount, test_flag, next_train_ep, train_ep, test_ep

    env_intr = 1
    EvalEpCount = SAC_EVAL_EP + 1
    test_flag = True
    next_train_ep = SAC_TRAIN_EP
    train_ep = 1
    test_ep = 1
    # ax = auxLib()

    sampling_time = 0
    rollout_time = 0
    training_time = 0
    mod1 = 0
    mod2 = 0
    mod3 = 0
    reset_st_time = now()
    reset_en_time = now()
    tot_runtime_2 = 0
    else_count = 0
    tot_runtime_list = [0]

    start_time = now()
    curr_runtime = 0
    pro_folder = os.getcwd()
    # Baseline ppo
    max_ac = 2000
    global route_keeper
    route_keeper = np.zeros(max_ac, dtype=int)

    if AGENT == "ppo_bl":
        # dir_name = "test_"+str(TEST_ID)+"_bl_"+AGENT
        dir_name = AGENT+"_"+MAP_ID
    else:
        # dir_name = "test_" + str(TEST_ID)+"_"+AGENT
        dir_name = AGENT+"_"+MAP_ID
    # ----- Log
    if LOAD_MODEL:
        lg = log(pro_folder + "/log/" + dir_name + "/log_inf"+".txt")
    else:
        lg = log(pro_folder + "/log/"+dir_name+"/log"+".txt")
    init()

    dt = syn_data(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    dt.display_data()

    cenv = countAirSim(dir_name=dir_name, pro_folder=pro_folder)

    route_keeper = cenv.init_env(route_keeper)


    # ----- Agents
    if AGENT == "random":
        agent = random_agent(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    elif AGENT == "min":
        agent = min_agent(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    elif AGENT == "max":
        agent = max_agent(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    elif "sac" in AGENT:
        print(" ############# SAC")
        from parameters import SAC_HID, SAC_L, SAC_GAMMA, SAC_EPOCHS, SAC_EXP_NAME
        from agents.sac.run_utils import setup_logger_kwargs

        # obs_ac_dim = dt.num_edges
        # obs_cr_dim = dt.num_edges + dt.num_edges*dt.num_los*dt.num_los*dt.eps_dim
        # act_dim = 1
        # obs_rw_dim = (dt.num_edges, dt.num_los * dt.num_los)
        # obs_rw_dim = (dt.num_edges, dt.num_los * dt.num_los * dt.eps_dim + dt.num_los * dt.num_los)
        # ntsk_dim = (dt.num_edges, dt.num_los, dt.num_los, dt.eps_dim)
        # rw_dim = (dt.num_edges, dt.num_los * dt.num_los)

        data_dir = pro_folder + "/log/"+dir_name
        logger_kwargs = setup_logger_kwargs(SAC_EXP_NAME, SEED, data_dir=data_dir)
        torch.set_num_threads(torch.get_num_threads())
        # agent = sac_global(actor_critic=core.MLPActorCritic,
        #     ac_kwargs=dict(hidden_sizes=[SAC_HID] * SAC_L),
        #     gamma=SAC_GAMMA, seed=SEED, epochs=SAC_EPOCHS,
        #     logger_kwargs=logger_kwargs,obs_ac_dim=obs_ac_dim, obs_cr_dim=obs_cr_dim, act_dim=act_dim, num_sectors = dt.num_edges, dir_name=dir_name, pro_folder=pro_folder, num_los=dt.num_los, eps_dim=dt.eps_dim, obs_rw_dim=obs_rw_dim, ntsk_dim=ntsk_dim, rw_dim=rw_dim)

        if "sac_dr_global" in AGENT:
            agent = sac_dr_global(actor_critic=core.MLPActorCritic,
    ac_kwargs = dict(hidden_sizes=[SAC_HID] * SAC_L),
    gamma = SAC_GAMMA, seed = SEED, epochs = SAC_EPOCHS,
    logger_kwargs = logger_kwargs, num_sectors = dt.num_edges, dir_name = dir_name, pro_folder = pro_folder, num_los = dt.num_los, eps_dim = dt.eps_dim, lr=LEARNING_RATE, replay_size=SAC_REPLAY_BUFFER)

        elif "sac_dr_indv" in AGENT:
            agent = sac_dr_indv(actor_critic=core.MLPActorCritic,
    ac_kwargs = dict(hidden_sizes=[SAC_HID] * SAC_L),
    gamma = SAC_GAMMA, seed = SEED, epochs = SAC_EPOCHS,
    logger_kwargs = logger_kwargs, num_sectors = dt.num_edges, dir_name = dir_name, pro_folder = pro_folder, num_los = dt.num_los, eps_dim = dt.eps_dim, lr=LEARNING_RATE, replay_size=SAC_REPLAY_BUFFER, lg=lg)

        elif "sac_random" in AGENT:
            agent = sac_random(actor_critic=core.MLPActorCritic,
                                ac_kwargs=dict(hidden_sizes=[SAC_HID] * SAC_L),
                                gamma=SAC_GAMMA, seed=SEED, epochs=SAC_EPOCHS,
                                logger_kwargs=logger_kwargs, num_sectors=dt.num_edges, dir_name=dir_name,
                                pro_folder=pro_folder, num_los=dt.num_los, eps_dim=dt.eps_dim, lr=LEARNING_RATE,
                                replay_size=SAC_REPLAY_BUFFER)

        elif "sac_rw_indv" in AGENT:
            agent = sac_rw_indv(actor_critic=core.MLPActorCritic,
    ac_kwargs = dict(hidden_sizes=[SAC_HID] * SAC_L),
    gamma = SAC_GAMMA, seed = SEED, epochs = SAC_EPOCHS,
    logger_kwargs = logger_kwargs, num_sectors = dt.num_edges, dir_name = dir_name, pro_folder = pro_folder, num_los = dt.num_los, eps_dim = dt.eps_dim, lr=LEARNING_RATE, replay_size=SAC_REPLAY_BUFFER)

    else:
        print("Error: Agent not found !")
        exit()

    start = False
    create_flag = True
    flag = True
    tot_reward = 0
    t = 0
    ep = 1
    lg.writeln("\n\n# ---------------------------------------------- #")
    lg.writeln("\nEpisode : "+str(ep))

    # ------ Buffers
    buff_nt = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los))
    buff_ntellv = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los, dt.eps_dim))
    # buff_ntellv = np.empty((0, dt.num_edges * dt.num_los * dt.num_los * dt.eps_dim))

    # buff_rt = np.zeros((dt.horizon, dt.num_edges))

    # global buff_rw_obs
    # rw_obs_dim = [dt.num_edges, dt.num_los*dt.num_los*dt.eps_dim + dt.num_los*dt.num_los]
    # rw_obs_dim = dt.num_edges * (dt.num_los * dt.num_los * dt.eps_dim + dt.num_los * dt.num_los)




    # buff_rw_obs = np.empty((0, rw_obs_dim))
    # buff_rt = np.empty((0, dt.num_edges))

    buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    buff_nm_conf = np.zeros(dt.horizon)
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    # ------- Mean Action
    mean_action = np.zeros(dt.num_edges)
    for e in range(dt.num_edges):
        mean_action[e] = np.mean(dt.action_space)

    # --------------- #
    global tmp_act_list, mu_list_train, std_list_train, mu_list_test, test_ep_counter
    test_ep_counter = 0
    tmp_act_list = {}
    mu_list_train = {}
    mu_list_test = {}
    std_list_train = {}
    for e in range(dt.num_edges):
        tmp_act_list[e] = []
        mu_list_train[e] = []
        mu_list_test[e] = []
        std_list_train[e] = []
    stack.stack('SEED ' + str(SEED))
    return config, stackfunctions

def update():

    global t, start, cenv, nt, flag, tot_reward, agent, pro_folder, ntellv
    global buff_nt, buff_ntellv, buff_rt,  buff_act_prob, ep, lg, mean_action
    global buff_nm_conf, route_keeper, dt, store_terminal, plot_conf, plot_goal_reached, plot_conf_mean, curr_runtime, start_time
    global sampling_time, rollout_time, training_time
    global env_intr, EvalEpCount, test_flag, train_ep, test_ep, mu_list_train, std_list_train, mu_list_test
    start = True
    step_time = now()

    if t == 0:
        cenv.avg_speed = {}
        cenv.avg_speed[-1] = []
        for e in range(cenv.num_edges):
            cenv.avg_speed[e] = []
        route_keeper = cenv.init_env(route_keeper)
        nt = cenv.init_state()
        ntellv = np.zeros((dt.num_edges,dt.num_los,dt.num_los,dt.eps_dim))

    o_ac = nt.sum(-1).sum(-1)
    num_ac = traf.ntraf
    indv_action = {}
    for i in range(num_ac):
        idx = traf.id[i]
        x, y = traf.lat[i], traf.lon[i]
        e = int(cenv.get_edges(x, y, cenv.alt))
        if e != -1:
            ac = agent.ac_list[e]
            x = np.array([o_ac.copy()])
            z_map = dt.zMap[e]
            x[:, [z_map]] = 0
            a, eps, mu, _ = agent.get_action(ac, x[0], deterministic=True)
            mu_list_test[e].append(mu)
            tmp_act_list[e].append(a)
            eps_ind = np.digitize([eps], dt.eps_bins)
            indv_action[idx] = (a, eps, eps_ind[0])
    t1 = now()
    ntp, ntellvp, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, indv_action=indv_action, route_keeper=route_keeper, agent=AGENT)
    sampling_time += getRuntime(t1, now())
    l1 = now()
    rt_sum = rt.sum()
    if VERBOSE:
        lg.writeln("\n   t : " + str(t) + " | rt : " + str(round(rt_sum, 3)))
    tot_reward += rt_sum
    nt = ntp.copy()
    ntellv = ntellvp.copy()

    # --------- Store Buffer
    buff_nt[t-1] = ntp
    buff_ntellv[t-1] = ntellv
    buff_nm_conf[t-1] = nm_conf
    # -------------------- #
    if t == HORIZON:
        t1 = now()
        # ------ Train
        t2 = now()
        training_time += getRuntime(t2, now())
        for i in range(traf.ntraf):
            idx = traf.id[i]
            stack.stack('DEL '+idx)
        # ------ Metrics
        # ------ Log
        avg_tr = round(np.mean(cenv.goal_time), 3)
        tot_cnf = buff_nm_conf.sum()
        test_ep = ep
        agent.log(ep, train_ep, test_ep, tot_reward, avg_tr, tot_cnf, test_flag, cenv.avg_speed, mu_list_train, std_list_train, mu_list_test)
        lg.writeln("\n\n   Test_Return : " + str(round(tot_reward, 3)))
        lg.writeln("   Test_Total_Conflicts : " + str(int(tot_cnf)))
        lg.writeln("   Test_Avg_Travel_Time : " + str(avg_tr) + "," + str(round(np.std(cenv.goal_time), 3)))
        lg.writeln("   Goal_Reached : " + str(cenv.goal_reached))
        rollout_time += getRuntime(t2, now())

        lg.writeln("   Sampling_Runtime : " + str(round(sampling_time, 3)))
        lg.writeln("   Rollout_Runtime : " + str(round(rollout_time, 3)))
        lg.writeln("   Training_Runtime : " + str(round(training_time, 3)))
        agent.writer.add_scalar('Runtime/Sampling_Time', sampling_time, ep)
        agent.writer.add_scalar('Runtime/Rollout_Time', rollout_time, ep)
        agent.writer.add_scalar('Runtime/Training_Time', training_time, ep)

        curr_runtime += getRuntime(t1, now())
        agent.writer.add_scalar('Runtime/Episode_Time', curr_runtime, ep)
        lg.writeln("   Episode_Runtime : " + str(round(curr_runtime, 3)))
        agent.writer.add_scalar('Runtime/Total_Time', round(getRuntime(
            start_time, now()), 3), ep)
        lg.writeln("   Total_Runtime : "+str(round(getRuntime(start_time, now()), 3)))

        curr_runtime = 0
        sampling_time = 0
        rollout_time = 0
        training_time = 0
        tot_reward = 0
        reset()
        return
    t += 1
    env_intr += 1

def preupdate():
    pass

def reset():
    # pass
    global start, cenv, flag, lg, buff_nt, buff_ntellv, buff_rt, route_keeper
    global plot_conf, plot_conf_mean, plot_goal_reached, reset_st_time, else_count, agent, reset_en_time, buff_act_prob, mean_action
    global ep, t#, # buff_rw_obs, buff_rt
    global EvalEpCount, next_train_ep, train_ep, test_ep, env_intr, test_ep_counter

    reset_st_time = now()
    ep += 1
    t = 0
    if ep == EPISODES+1:
        stack.stack('STOP')
        stack.stack('SEED '+str(SEED))

    lg.writeln("# ---------------------------------------------- #")
    lg.writeln("\nEpisode : "+str(ep))
    flag = True

    # ------ Clear Buffers
    buff_nt = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los))
    buff_ntellv = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los, dt.eps_dim))

    buff_rt = np.zeros((dt.horizon, dt.num_edges))
    buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    cenv.goal_reached = 0
    cenv.goal_time = []
    cenv.abs_num_ac = 0 #cenv.num_env_ac
    reset_en_time = now()

    global tmp_act_list, mu_list_train, std_list_train, mu_list_test
    tmp_act_list = {}
    mu_list_train = {}
    std_list_train = {}
    mu_list_test = {}
    for e in range(dt.num_edges):
        tmp_act_list[e] = []
        mu_list_train[e] = []
        std_list_train[e] = []
        mu_list_test[e] = []
    stack.stack('IC count.scn')

def init():

    global dir_name, pro_folder
    os.system("mkdir "+pro_folder+"/log/")
    os.system("mkdir "+pro_folder+"/log/" + dir_name)
    if LOAD_MODEL is False:
        deleteDir(pro_folder+"/log/"+dir_name + "/plots/")
    os.system("cp "+pro_folder+"/"+"parameters.py "+pro_folder+"/log/"+dir_name+"/")
    os.system("mkdir "+pro_folder+"/log/"+dir_name+"/plots")
    os.system("mkdir "+pro_folder+"/log/" + dir_name + "/model")

