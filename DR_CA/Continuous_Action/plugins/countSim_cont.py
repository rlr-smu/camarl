""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, scr, tools
from parameters import EPISODES, HORIZON, VERBOSE, SEED, BATCH_SIZE, SHOULD_LOG, SAVE_MODEL, LOAD_MODEL, AGENT, MAP_ID, TEST_ID, UPDATE_INTV
from count_env_cont import countAirSim
from ipdb import set_trace
from data import syn_data
from agents.random_agent import random_agent, min_agent, max_agent
from agents.mtmf_local_act_cont import mtmf_local_act_cont
from utils import display, log, deleteDir, real_time
import numpy as np
import os
from Multi_Agent.PPO import PPO_Agent
from Multi_Agent.PPO import getClosestAC
import matplotlib.pyplot as plt
from auxLib3 import now, getRuntime
import time
# plt.ion()

def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name':     'countSim_cont',
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
        dir_name = AGENT+"_"+MAP_ID
    else:
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
    elif AGENT == "mtmf_local_act_cont":
        agent = mtmf_local_act_cont(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
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
    buff_rt = np.zeros((dt.horizon, dt.num_edges))
    # buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    buff_nm_conf = np.zeros(dt.horizon)
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    # ------- Mean Action
    mean_action = np.zeros(dt.num_edges)
    # for e in range(dt.num_edges):
    #     mean_action[e] = np.mean(dt.action_space)
    # --------------- #
    stack.stack('SEED ' + str(SEED))

    return config, stackfunctions

def update():

    global t, start, cenv, nt, flag, tot_reward, agent, pro_folder
    global buff_nt, buff_ntellv, buff_rt,  buff_act_prob, ep, lg, mean_action
    global buff_nm_conf, route_keeper, dt, store_terminal, plot_conf, plot_goal_reached, plot_conf_mean, curr_runtime, start_time
    global sampling_time, rollout_time, training_time
    start = True
    step_time = now()

    # if flag:
    #     flag = False
    #     t = 1
    #     route_keeper = cenv.init_env(route_keeper)
    #     nt = cenv.init_state()
    #     init_time = getRuntime(step_time, now())
    # lg.writeln(str(t) + "," + str(traf.ntraf))

    if t == 0:
        route_keeper = cenv.init_env(route_keeper)
        nt = cenv.init_state()

    # if VERBOSE:
    #     display(lg, nt)
    # -------------------- #

    if AGENT == "ppo_bl":
        store_terminal = np.zeros(len(traf.id), dtype=int)
        for i in range(len(traf.id)):
            T, type = agent.update(traf, i, route_keeper)
        n_ac = len(traf.id)
        if n_ac > 0:
            ind = np.array([int(x[2:]) for x in traf.id])
            route = route_keeper[ind]
            state_ppo = np.zeros((n_ac,5))
            state_ppo[:,0] = traf.lat
            state_ppo[:,1] = traf.lon
            state_ppo[:,2] = traf.tas
            state_ppo[:,3] = route
            state_ppo[:,4] = traf.ax
            state_ppo, context = getClosestAC(state_ppo, traf, store_terminal, agent)
            action_ac = agent.get_action(state_ppo, context)
        else:
            action_ac = None
        # dummy prob, not used
        # action_prob = np.random.uniform(1, 10, size=(dt.num_edges, dt.num_actions))
        # for e in range(dt.num_edges):
        #     action_prob[e] = action_prob[e] / action_prob[e].sum()
        ntp, ntellv, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, action_prob=None, route_keeper=route_keeper, action_ac=action_ac, agent=AGENT)


    elif "mtmf" in AGENT:
        n_ac = len(traf.id)
        if n_ac > 0:
            mean_action = agent.update_mean_action(nt, cenv.edge_ac, mean_action)
            action_ac = agent.get_action(t, nt, mean_action, cenv.ac_edge)
        else:
            action_ac = None

        # dummy prob, not used
        # action_prob = np.random.uniform(1, 10, size=(dt.num_edges, dt.num_actions))
        # for e in range(dt.num_edges):
        #     action_prob[e] = action_prob[e] / action_prob[e].sum()

        ntp, ntellv, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, action_prob=None, route_keeper=route_keeper, action_ac=action_ac, agent=AGENT)
        next_mean_action = agent.update_mean_action(ntp, cenv.edge_ac, mean_action)
        agent.get_next_state_train(t, ntp, next_mean_action, cenv.ac_edge)

    else:
        t1 = now()
        indv_action = agent.get_action(nt)
        ntp, ntellv, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, indv_action=indv_action, route_keeper=route_keeper, agent=AGENT)
        sampling_time += getRuntime(t1, now())


    l1 = now()
    rt_sum = rt.sum()
    if VERBOSE:
        lg.writeln("\n   t : " + str(t) + " | rt : " + str(round(rt_sum, 3)))
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(traf.gs)
    # print(traf.id)
    # print(traf.lat)
    # print(traf.lon)


    tot_reward += rt_sum
    nt = ntp.copy()

    # --------- Store Buffer
    buff_nt[t-1] = ntp
    buff_ntellv[t-1] = ntellv
    buff_rt[t-1] = rt
    # buff_act_prob[t-1] = action_prob
    buff_nm_conf[t-1] = nm_conf

    # ---------- Real time plot
    plot_conf.append(nm_conf)
    mn = np.mean(plot_conf)
    plot_conf_mean.append(mn)
    plot_goal_reached.append(cenv.goal_reached)
    curr_runtime += getRuntime(step_time, now())

    # -------------------- #
    if t == HORIZON:

        t1 = now()
        # ------ Store
        t2 = now()

        agent.store_rollouts(buff_nt=buff_nt, buff_ntellv=buff_ntellv, buff_rt=buff_rt, buff_act_prob=None)


        #t1 = np.array(agent.diff_ret_tmp)
        #set_trace()
        # np.save(pro_folder+"/log/"+dir_name +"/dr", np.array(agent.diff_ret_tmp))


        # ------ Log
        lg.writeln("\n\n   Return : "+ str(round(tot_reward, 3)))
        rollout_time += getRuntime(t2, now())

        # ------ Train
        t2 = now()
        if ep % BATCH_SIZE == 0:
            if LOAD_MODEL == False:
                agent.train(ep=ep)
            lg.writeln("   Loss : " + str(round(agent.loss, 3)))
            agent.clear_buffer()

        # ------ Save model
        if ep % SAVE_MODEL == 0 and LOAD_MODEL == False:
            agent.save_model(tot_reward, ep)

        training_time += getRuntime(t2, now())

        for i in range(traf.ntraf):
            idx = traf.id[i]
            stack.stack('DEL '+idx)

        # ------ Metrics
        avg_tr = round(np.mean(cenv.goal_time), 3)
        avg_cnf = round(buff_nm_conf.mean(), 3)
        # lg.writeln("   Avg. Conflicts : " + str(avg_cnf)+","+str(round(buff_nm_conf.std(), 3)))
        # lg.writeln("   Max Conflicts : " + str(round(buff_nm_conf.max(), 3)))
        tot_cnf = buff_nm_conf.sum()
        lg.writeln("   Total_Conflicts : " + str(int(tot_cnf)))
        lg.writeln("   Goal_Reached : " + str(cenv.goal_reached))
        lg.writeln("   Avg_Travel_Time : "+str(avg_tr)+","+str(round(np.std(cenv.goal_time), 3)))


        tmp = buff_ntellv.sum(2).sum(2)
        mean_act_count = tmp[:, :, :].mean(0)

        if VERBOSE:
            lg.writeln("\n   Avg. Action count : ")
            lg.writeln("   "+ str(mean_act_count))


        # for e in range(mean_act_count.shape[0]):
        #     y = mean_act_count[e]
        #     y = list(y)
        #     plt.xticks([i for i in range(len(y))])
        #     plt.ylabel("Avg. Count")
        #     plt.xlabel("Noise partition")
        #     plt.bar([i for i in range(len(y))], y)
        #     plt.show()
        # set_trace()


        if ep % SHOULD_LOG == 0:
            agent.log(ep, tot_reward, None, avg_tr, avg_cnf, tot_cnf, cenv.goal_reached, mean_act_count)


        lg.writeln("   Sampling Runtime : " + str(round(sampling_time, 3)))
        lg.writeln("   Rollout Runtime : " + str(round(rollout_time, 3)))
        lg.writeln("   Training Runtime : " + str(round(training_time, 3)))

        agent.writer.add_scalar('Runtime/Sampling_Time', sampling_time, ep)
        agent.writer.add_scalar('Runtime/Rollout_Time', rollout_time, ep)
        agent.writer.add_scalar('Runtime/Training_Time', training_time, ep)

        curr_runtime += getRuntime(t1, now())
        agent.writer.add_scalar('Runtime/Episode_Time', curr_runtime, ep)
        lg.writeln("   Episode Runtime : " + str(round(curr_runtime, 3)))


        agent.writer.add_scalar('Runtime/Total_Time', round(getRuntime(
            start_time, now()), 3), ep)
        lg.writeln("   Total Runtime : "+str(round(getRuntime(start_time, now()), 3)))

        curr_runtime = 0
        sampling_time = 0
        rollout_time = 0
        training_time = 0
        tot_reward = 0
        reset()
        return
    t += 1

def preupdate():
    pass

def reset():
    # pass
    global start, cenv, flag, lg, buff_nt, buff_ntellv, buff_rt, route_keeper
    global plot_conf, plot_conf_mean, plot_goal_reached, reset_st_time, else_count, agent, reset_en_time, buff_act_prob, mean_action
    global ep, t

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
    # buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    # ------- Mean Action
    mean_action = np.zeros(dt.num_edges)
    # for e in range(dt.num_edges):
    #     mean_action[e] = np.mean(dt.action_space)

    cenv.goal_reached = 0
    cenv.goal_time = []
    cenv.abs_num_ac = 0 #cenv.num_env_ac
    reset_en_time = now()
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

