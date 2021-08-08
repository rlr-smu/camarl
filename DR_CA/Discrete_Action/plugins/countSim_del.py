""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, scr, tools
from parameters import EPISODES, HORIZON, VERBOSE, SEED, BATCH_SIZE, SHOULD_LOG, SAVE_MODEL, TEST_ID, LOAD_MODEL, AGENT, MAP_ID
from count_env_del import countAirSim_del
from ipdb import set_trace
from data import syn_data
# from agents.vpg import vpg
# from agents.diff_rw_mean_nbr import diff_rw_mean_nbr
# from agents.diff_rw_mean_act import diff_rw_mean_act
# from agents.diff_rw_v2 import diff_rw_v2
# from agents.diff_rw_v2_grad import diff_rw_v2_grad

from agents.random_agent import random_agent, min_agent, max_agent
from agents.vpg_sep import vpg_sep
from agents.baseline_nn import vpg_bl_multi, vpg_bl_single
from agents.diff_rw_v2_grad_sep import diff_rw_v2_grad_sep
from agents.diff_rw_v2_grad_sep_mid import diff_rw_v2_grad_sep_mid
from agents.diff_rw_ana import diff_rw_ana
from agents.ind_lrn import ind_lrn
from agents.diff_ppo_mid import diff_ppo_mid
from agents.diff_ppo_max import diff_ppo_max
from utils import display, log, deleteDir, real_time
import numpy as np
import os
from Multi_Agent.PPO import PPO_Agent
from Multi_Agent.PPO import getClosestAC
import matplotlib.pyplot as plt
from auxLib3 import now, getRuntime
import time
plt.ion()

def init_plugin():

    # Configuration parameters
    config = {
        'plugin_name':     'countSim_del',
        'plugin_type':     'sim',
        'update_interval': 12.0,
        'update':          update
        #'preupdate':       preupdate,
        #'reset':         reset
        }
    stackfunctions = {     }

    # ------- RL Environment
    global cenv, nt, flag, ep, t, start, create_flag, tot_reward, dt, agent
    global buff_nt, buff_ntellv, buff_rt, dir_name, lg, buff_act_prob
    global buff_nm_conf, pro_folder, plot_conf, plot_goal_reached, plot_conf_mean
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
    cenv = countAirSim_del(dir_name=dir_name, pro_folder=pro_folder)

    # ----- Agents
    if AGENT == "random":
        agent = random_agent(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    elif AGENT == "min":
        agent = min_agent(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "max":
        agent = max_agent(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "vpg_sep":
        agent = vpg_sep(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "vpg_bl_single":
        agent = vpg_bl_single(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    elif AGENT == "vpg_bl_multi":
        agent = vpg_bl_multi(dir_name=dir_name, pro_folder=pro_folder, lg=lg)
    elif AGENT == "diff_rw_v2_grad_sep":
        agent = diff_rw_v2_grad_sep(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "diff_rw_v2_grad_sep_mid":
        agent = diff_rw_v2_grad_sep_mid(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "diff_rw_ana":
        agent = diff_rw_ana(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "ind_lrn":
        agent = ind_lrn(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "diff_ppo_mid":
        agent = diff_ppo_mid(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "diff_ppo_max":
        agent = diff_ppo_max(dir_name=dir_name, pro_folder=pro_folder, lg=lg)

    elif AGENT == "ppo_bl":
        # Baseline
        max_ac = dt.max_ac
        n_states = 5
        # positions = np.array([[ 40.2, -95.2,  38.6, -92.,  79],[ 38.6, -95.2,  40.2, -92.,  97]])
        positions = np.load("./routes_james/"+dt.map_id+"/" + dt.map_id + "_lines.npy", allow_pickle=True)
        num_intruders = 4
        agent = PPO_Agent(n_states,3,positions.shape[0],EPISODES,positions,num_intruders,load_model=LOAD_MODEL, dir_name=dir_name)

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
    buff_ntellv = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los, dt.num_actions))
    buff_rt = np.zeros((dt.horizon, dt.num_edges))
    buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    buff_nm_conf = np.zeros(dt.horizon)
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    # --------------- #
    route_keeper = cenv.init_env(route_keeper)
    stack.stack('SEED ' + str(SEED))


    return config, stackfunctions

def update():

    global t, start, cenv, nt, flag, tot_reward, agent
    global buff_nt, buff_ntellv, buff_rt,  buff_act_prob, ep, lg
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

    if t == 0:
        route_keeper = cenv.init_env(route_keeper)
        nt = cenv.init_state()



    if VERBOSE:
        display(lg, nt)
    # -------------------- #

    if AGENT == "ppo_bl":
        store_terminal = np.zeros(len(traf.id), dtype=int)
        for i in range(len(traf.id)):
            T, type = agent.update(traf, i, route_keeper)
            # if T and type == 2:
            #     store_terminal[i] = 1
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
        # dummy prob
        action_prob = np.random.uniform(1, 10, size=(dt.num_edges, dt.num_actions))
        for e in range(dt.num_edges):
            action_prob[e] = action_prob[e] / action_prob[e].sum()
        # ntp, ntellv, rt, nm_conf, _, route_keeper, ntellv_mean = cenv.step(ep=ep, t=t, state=nt, action_prob=action_prob, route_keeper=route_keeper, action_ac=action_ac, agent=AGENT)
        ntp, ntellv, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, action_prob=action_prob, route_keeper=route_keeper, action_ac=action_ac, agent=AGENT)
    else:
        t1 = now()
        action_prob = agent.get_action(nt)
        ntp, ntellv, rt, nm_conf, _, route_keeper = cenv.step(ep=ep, t=t, state=nt, action_prob=action_prob, route_keeper=route_keeper, agent=AGENT)
        sampling_time += getRuntime(t1, now())


    l1 = now()
    rt_sum = rt.sum()
    lg.writeln("\n   t : " + str(t) + " | rt : " + str(round(rt_sum, 3)))

    # lg.writeln("\n   n_ac : " + str(traf.ntraf)+" "+str(traf.id))
    # lg.writeln(str(t) + "," + str(traf.id)+ str(round(rt_sum, 3)))

    tot_reward += rt_sum
    nt = ntp.copy()

    # --------- Store Buffer
    buff_nt[t-1] = ntp
    buff_ntellv[t-1] = ntellv
    buff_rt[t-1] = rt
    buff_act_prob[t-1] = action_prob
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
        agent.store_rollouts(buff_nt=buff_nt, buff_ntellv=buff_ntellv, buff_rt=buff_rt, buff_act_prob=buff_act_prob)

        # ------ Log
        lg.writeln("\n\n   Total Reward : "+ str(round(tot_reward, 3)))
        rollout_time += getRuntime(t2, now())

        # ------ Train
        t2 = now()
        if ep % BATCH_SIZE == 0:
            if LOAD_MODEL == False:
                agent.train(ep=ep)

            agent.clear_buffer()

        # ------ Save model
        if ep % SAVE_MODEL == 0:
            agent.save_model(tot_reward, ep)

        training_time += getRuntime(t2, now())

        for i in range(traf.ntraf):
            idx = traf.id[i]
            stack.stack('DEL '+idx)

        # ------ Metrics
        avg_tr = round(np.mean(cenv.goal_time), 3)
        avg_cnf = round(buff_nm_conf.mean(), 3)
        tot_cnf = buff_nm_conf.sum()
        lg.writeln("   Avg. Conflicts : " + str(avg_cnf)+","+str(round(buff_nm_conf.std(), 3)))
        lg.writeln("   Max Conflicts : " + str(round(buff_nm_conf.max(), 3)))
        lg.writeln("   Total  Conflicts : " + str(int(tot_cnf)))
        tot_cnf = buff_nm_conf.sum()

        lg.writeln("   Total Conflicts : " + str(int(tot_cnf)))
        lg.writeln("   Goal Reached : " + str(cenv.goal_reached))
        lg.writeln("   Avg. Travel Time : "+str(avg_tr)+","+str(round(np.std(cenv.goal_time), 3)))
        tmp = buff_ntellv.sum(2).sum(2)
        mean_count = tmp[:, :, :].mean(0)
        lg.writeln("\n   Avg. Action count : ")
        lg.writeln("   "+ str(mean_count))

        if ep % SHOULD_LOG == 0:
            agent.log(ep, tot_reward, buff_act_prob, avg_tr, avg_cnf, tot_cnf, cenv.goal_reached)


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
    global plot_conf, plot_conf_mean, plot_goal_reached, reset_st_time, else_count, agent, reset_en_time, buff_act_prob
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
    buff_ntellv = np.zeros((dt.horizon, dt.num_edges, dt.num_los, dt.num_los, dt.num_actions))
    buff_rt = np.zeros((dt.horizon, dt.num_edges))
    buff_act_prob = np.zeros((dt.horizon, dt.num_edges, dt.num_actions))
    plot_conf = []
    plot_goal_reached = []
    plot_conf_mean = []

    cenv.goal_reached = 0
    cenv.goal_time = []
    cenv.abs_num_ac = 0
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

