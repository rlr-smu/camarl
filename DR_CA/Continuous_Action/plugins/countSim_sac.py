""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, scr, tools
from bluesky.tools.areafilter import checkInside, areas

from parameters import EPISODES, HORIZON, VERBOSE, SEED, EP_BATCH_SIZE, SHOULD_LOG, SAVE_MODEL, LOAD_MODEL, AGENT, MAP_ID, TEST_ID, UPDATE_INTV, SAC_UPDATE_AFTER, SAC_UPDATE_EVERY, SAC_BATCH_SIZE, SAC_WARMUP, SAC_TRAIN_EPOCH, LEARNING_RATE, SAC_REPLAY_BUFFER, SAC_TEST_EVERY, SAC_EVAL_EP, SAC_TRAIN_EP
from count_env_cont import countAirSim


from ipdb import set_trace
from data import syn_data


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
        'plugin_name':     'countSim_sac',
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
    test_flag = False
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
    if "sac" in AGENT:
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

    if t == HORIZON:
        done = True
    else:
        done = False

    # if ep % SAC_TEST_EVERY == 0:
    # # if ep > SAC_TEST_EVERY:
    #     test_flag = True
    #     evalEpCount
    # else:
    #     test_flag = False

    if t == 0:
        cenv.avg_speed = {}
        cenv.avg_speed[-1] = []
        for e in range(cenv.num_edges):
            cenv.avg_speed[e] = []
        route_keeper = cenv.init_env(route_keeper)
        nt = cenv.init_state()
        ntellv = np.zeros((dt.num_edges,dt.num_los,dt.num_los,dt.eps_dim))

    # lg.writeln(str(ep)+","+ str(t)+ ","+str(test_flag))

    # if VERBOSE:
    #     display(lg, nt)
    # -------------------- #
    obs_ac = {}
    act = {}
    for z in range(dt.num_edges):
        obs_ac[z] = np.empty((0, dt.num_edges))
        act[z] = np.empty((0, 1))

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
            if env_intr < SAC_WARMUP:
                a, eps = agent.get_random_action()
            else:
                if test_flag:
                    a, eps, mu, _ = agent.get_action(ac, x[0], deterministic=True)
                    mu_list_test[e].append(mu)
                else:
                    a, eps, mu, std = agent.get_action(ac, x[0])
                    mu_list_train[e].append(mu)
                    std_list_train[e].append(std)
                tmp_act_list[e].append(a)
            obs_ac[e] = np.vstack((obs_ac[e], x[0]))
            act[e] = np.vstack((act[e], a))
            eps_ind = np.digitize([eps], dt.eps_bins)
            indv_action[idx] = (a, eps, eps_ind[0])

    # --- Diff Reward
    dr_z, rw_obs = agent.compute_diff_rw(nt, ntellv, dt.ntsk_indx, dt.mid_nts, dt.mid_ntsk)

    # Add replay buffer for reward
    if test_flag == False:
        rw_scaled = dt.rw_scale * dt.rw_train_z
        agent.replay_buffer_rw.store(rw_obs, ntellv, rw_scaled)
    t1 = now()
    ntp, ntellvp, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, indv_action=indv_action, route_keeper=route_keeper, agent=AGENT)
    sampling_time += getRuntime(t1, now())
    obs2_ac = {}
    obs_cr = {}
    obs2_cr = {}
    o2_ac = ntp.sum(-1).sum(-1)
    for z in range(dt.num_edges):
        # --- o2_ac
        x = np.array([o2_ac.copy()])
        z_map = dt.zMap[z]
        x[:, [z_map]] = 0
        z_count = obs_ac[z].shape[0]
        xp = np.tile(x, (z_count, 1))
        obs2_ac[z] = xp

        # --- o_cr
        x_cr = agent.prepare_obs_cr(ntellv, z, z_count)
        x_ac = obs_ac[z]
        obs_cr[z] = np.hstack((x_ac, x_cr))

        # --- o2_cr
        xp_cr = agent.prepare_obs_cr(ntellvp, z, z_count)
        # xp_cr = np.array(o2_cr.copy())
        # xp_cr = np.tile(xp_cr, (z_count, 1))
        xp_ac = obs2_ac[z]
        obs2_cr[z] = np.hstack((xp_ac, xp_cr))

        # --- User DR signal
        if "rw_indv" in AGENT:
            rew = rt.sum()
        else:
            rew = dr_z[z]

        if test_flag == False:
            for i in range(z_count):
                agent.replay_buffer_list[z].store(obs_ac[z][i], obs_cr[z][i], act[z][i], rew, obs2_ac[z][i], obs2_cr[z][i], done)

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

    # ------- SAC UPDATE
    if env_intr % SAC_UPDATE_EVERY == 0 and test_flag == False and env_intr > SAC_UPDATE_AFTER:
        for _ in range(SAC_TRAIN_EPOCH):
            agent.update_rw(SAC_BATCH_SIZE)
            agent.update(SAC_BATCH_SIZE)

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
        agent.log(ep, train_ep, test_ep, tot_reward, avg_tr, tot_cnf, test_flag, cenv.avg_speed, mu_list_train, std_list_train, mu_list_test)


        if test_flag:
            lg.writeln("\n\n   Test_Return : " + str(round(tot_reward, 3)))
            lg.writeln("   Test_Total_Conflicts : " + str(int(tot_cnf)))
            lg.writeln("   Test_Avg_Travel_Time : " + str(avg_tr) + "," + str(round(np.std(cenv.goal_time), 3)))
            lg.writeln("   Goal_Reached : " + str(cenv.goal_reached))
            agent.curr_tcnf_list.append(tot_cnf)
            if EvalEpCount == SAC_EVAL_EP:
                mean_curr_tcnf = np.mean(agent.curr_tcnf_list)
                if agent.tcnf_min > mean_curr_tcnf:
                    agent.save_model(test_ep)
                    agent.tcnf_min = mean_curr_tcnf
                agent.curr_tcnf_list = []
        else:
            lg.writeln("\n\n   Train_Return : "+ str(round(tot_reward, 3)))
            lg.writeln("   Train_Total_Conflicts : " + str(int(tot_cnf)))
            lg.writeln("   Train_Avg_Travel_Time : " + str(avg_tr) + "," + str(round(np.std(cenv.goal_time), 3)))
            lg.writeln("   Goal_Reached : " + str(cenv.goal_reached))
            if env_intr > SAC_WARMUP:
                lg.writeln("   Loss_rw : " + str(round(agent.loss_rw, 3)))
                lg.writeln("   Loss_q : " + str(round(agent.loss_q, 3)))
                lg.writeln("   Loss_pi : " + str(round(agent.loss_p, 3)))


        rollout_time += getRuntime(t2, now())
        dtpt = buff_ntellv.shape[0]
        buff_ntellv = buff_ntellv.reshape(dtpt, dt.num_edges, dt.num_los,  dt.num_los, dt.eps_dim)

        tmp = buff_ntellv.sum(2).sum(2)
        mean_act_count = tmp[:, :, :].mean(0)

        if VERBOSE:
            lg.writeln("\n   Avg_Action_count : ")
            lg.writeln("   "+ str(mean_act_count))

        # buff_ntellv = np.empty((0, dt.num_edges * dt.num_los * dt.num_los * dt.eps_dim))


        # if ep % SHOULD_LOG == 0:
        #     agent.log(ep, tot_reward, avg_tr, avg_cnf, tot_cnf, cenv.goal_reached)


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
    if test_flag == False:
        env_intr += 1

def preupdate():
    pass

def reset():
    # pass
    global start, cenv, flag, lg, buff_nt, buff_ntellv, buff_rt, route_keeper
    global plot_conf, plot_conf_mean, plot_goal_reached, reset_st_time, else_count, agent, reset_en_time, buff_act_prob, mean_action
    global ep, t#, # buff_rw_obs, buff_rt
    global EvalEpCount, test_flag, next_train_ep, train_ep, test_ep, env_intr, test_ep_counter


    reset_st_time = now()
    ep += 1
    t = 0

    # ---- Test Criteria
    if env_intr > SAC_WARMUP:
        if EvalEpCount < SAC_EVAL_EP or ep == next_train_ep:
            test_flag = True
            if ep == next_train_ep:
                EvalEpCount = 0
                next_train_ep = ep + SAC_TRAIN_EP + SAC_EVAL_EP
                #test_ep_counter = 0
            else:
                EvalEpCount += 1
            test_ep += 1
        else:
            test_flag = False
            train_ep += 1
    else:
        test_flag = False


    if ep == EPISODES+1:
        stack.stack('STOP')
        stack.stack('SEED '+str(SEED))
    lg.writeln("# ---------------------------------------------- #")
    lg.writeln("\nEpisode : "+str(ep))
    flag = True

    # ------ Clear Buffers
    buff_nt = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los))
    buff_ntellv = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los, dt.eps_dim))
    # buff_ntellv = np.empty((0, dt.num_edges * dt.num_los * dt.num_los * dt.eps_dim))

    buff_rt = np.zeros((dt.horizon, dt.num_edges))
    buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    #rw_obs_dim = dt.num_edges * (dt.num_los * dt.num_los * dt.eps_dim + dt.num_los * dt.num_los)
    # buff_rw_obs = np.empty((0, rw_obs_dim))
    #buff_rt = np.empty((0, dt.num_edges))

    # ------- Mean Action
    mean_action = np.zeros(dt.num_edges)
    for e in range(dt.num_edges):
        mean_action[e] = np.mean(dt.action_space)

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

