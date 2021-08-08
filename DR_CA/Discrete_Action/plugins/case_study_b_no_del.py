""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from bluesky import navdb
from bluesky.tools.aero import ft
from bluesky.tools import geo, areafilter
from Multi_Agent.PPO_no_del import PPO_Agent
import geopy.distance
import tensorflow as tf
import random
import pandas as pd
from operator import itemgetter
from shapely.geometry import LineString
import numba as nb
import time
from ipdb import set_trace
from torch.utils.tensorboard import SummaryWriter
import os
from parameters import HORIZON, LOAD_MODEL, TEST_ID, EPISODES, MAP_ID, MAX_AC, ARR_INTERVAL, SEED, MAX_WIND, WIND, ACTIONS
from utils import display, log, deleteDir
from auxLib3 import loadDataStr
from functools import reduce
from bluesky.tools.areafilter import checkInside, areas
## For running on GPU
# from keras.backend.tensorflow_backend import set_session
# from shapely.geometry import LineString
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# sess = tf.Session(config=config)
# set_session(sess)




### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.

def init_plugin():

    global num_ac
    global counter
    global ac
    global max_ac
    global positions
    global agent
    global best_reward
    global num_success
    global success
    global collisions
    global num_collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start
    global writer
    global dir_name
    global lg
    global pro_folder
    global directions
    global num_entry
    global goal_time
    global collisions_list
    global total_reward


    # ------- Seed -------- #
    stack.stack('SEED '+str(SEED))
    np.random.seed(SEED)
    random.seed(SEED)


    pro_folder = os.getcwd()

    num_success_train = []
    num_collisions_train = []

    num_success = []
    num_collisions = []
    previous_action = {}
    last_observation = {}
    observation = {}
    collisions = 0
    success = 0
    num_intruders = 4
    goal_time = []
    collisions_list = []
    total_reward = 0

    # ----- Log
    dir_name = "ppo_bl_no_del" + "_" + MAP_ID
    if LOAD_MODEL:
        lg = log(pro_folder + "/log/" + dir_name + "/log_inf.txt")
    else:
        lg = log(pro_folder + "/log/"+dir_name+"/log.txt")
    init()
    writer = SummaryWriter(pro_folder+"/log/"+dir_name+"/plots/")


    num_ac = 0
    max_ac = MAX_AC
    best_reward = -10000000
    ac_counter = 0
    n_states = 5
    route_keeper = np.zeros(max_ac,dtype=int)

    # positions = np.load('./routes/case_study_b_route.npy')
    positions = np.load("./routes_james/"+MAP_ID+"/"+MAP_ID+"_lines.npy", allow_pickle=True)
    directions = loadDataStr("./routes_james/"+MAP_ID+"/"+MAP_ID+"_directions")
    num_entry = len(directions.keys())


    # Poly
    routes_list = add_poly()


    choices = ARR_INTERVAL
    # choices = [3, 10, 15]
    # route_queue = random.choices(choices,k=positions.shape[0])
    route_queue = random.choices(choices, k=num_entry)
    # set_trace()

    actions_size = len(ACTIONS)

    # agent = PPO_Agent(n_states,actions_size,num_entry,EPISODES,positions,num_intruders, dir_name=dir_name, actions=ACTIONS, routes_list=routes_list)

    agent = PPO_Agent(n_states,actions_size,num_entry,EPISODES,positions,num_intruders, dir_name=dir_name, actions=ACTIONS, routes_list=routes_list)


    counter = 0
    start = time.time()

    # Addtional initilisation code
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'CASE_STUDY_B_NO_DEL',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 12.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.

        'update':      update}

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        #'reset':         reset
        #}
    stackfunctions = {
        }

    return config, stackfunctions

def update():
    """given a current state in the simulation, allow the agent to select an action.
     "" Then send the action to the bluesky command line ""
    """
    global num_ac
    global counter
    global ac
    global max_ac
    global positions
    global directions
    global agent
    global success
    global collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global choices
    global start
    global num_entry
    global directions
    global goal_time
    global collisions_list
    global total_reward

    store_terminal = {}

    agent.rt = 0


    # if counter == 0:
    if agent.episode_count > 0:
        add_poly()
        #counter = 1
        #return
        # set_trace()

    # ----- Add wind
    if WIND:
        add_wind()

    if ac_counter < max_ac:  ## maybe spawn a/c based on time, not based on this update interval

        # print(">>>>>>", agent.episode_count, counter, ac_counter)

        if ac_counter == 0:

            for did in range(num_entry):
                i = np.random.choice(directions[did])
                wps = positions[i][:-1]
                hdg = positions[i][-1]
                o_lat = wps[0]
                o_lon = wps[1]
                g_lat = wps[-2]
                g_lon = wps[-1]
                stack.stack('CRE KL{}, A320, {}, {}, {}, 2000, 100'.format(ac_counter, o_lat, o_lon, hdg))
                j = 2
                while j < len(wps)-2:
                    lat = wps[j]
                    lon = wps[j + 1]
                    j += 2
                    stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter,lat,lon))
                stack.stack('DEST KL{}, {}, {}'.format(ac_counter, g_lat, g_lon))
                # for i in range(len(positions)):
                # lat,lon,glat,glon,h = positions[i]
                # stack.stack('CRE KL{}, A320, {}, {}, {}, 2000, 100'.format(ac_counter,lat,lon,h))
                # stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter,glat,glon))
                route_keeper[ac_counter] = i
                num_ac += 1
                ac_counter += 1

                # print(positions)
                # exit()
                # set_trace()

        else:
            for k in range(len(route_queue)):
                if counter == route_queue[k]:
                    i = np.random.choice(directions[k])
                    wps = positions[i][:-1]
                    hdg = positions[i][-1]
                    o_lat = wps[0]
                    o_lon = wps[1]
                    g_lat = wps[-2]
                    g_lon = wps[-1]
                    stack.stack('CRE KL{}, A320, {}, {}, {}, 2000, 100'.format(ac_counter, o_lat, o_lon, hdg))
                    j = 2
                    while j < len(wps)-2:
                        lat = wps[j]
                        lon = wps[j + 1]
                        j += 2
                        stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter, lat, lon))
                    stack.stack('DEST KL{}, {}, {}'.format(ac_counter, g_lat, g_lon))
                    # lat,lon,glat,glon,h = positions[k]
                    # stack.stack('CRE KL{}, A320, {}, {}, {}, 2000, 100'.format(ac_counter,lat,lon,h))
                    # stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter,glat,glon))
                    route_keeper[ac_counter] = i
                    num_ac += 1
                    ac_counter += 1
                    tmp = random.choices(choices,k=1)[0]
                    route_queue[k] = counter + tmp
                    if ac_counter == max_ac:
                        break

    store_terminal = np.zeros(len(traf.id),dtype=int)
    col_step = 0
    for i in range(len(traf.id)):
        T,type_ = agent.update(traf,i,route_keeper)
        id_ = traf.id[i]
        if T:
            # stack.stack('DEL {}'.format(id_))
            # num_ac -=1
            if type_ == 1:
                collisions += 1
                col_step += 1

            if type_ == 2:
                # set_trace()
                stack.stack('DEL {}'.format(id_))
                num_ac -=1
                goal_time.append(traf.timeflown[i])
                success += 1

                store_terminal[i] = 1
                if id_ in  previous_action:
                    agent.store(last_observation[id_],previous_action[id_],[np.zeros(last_observation[id_][0].shape),np.zeros(last_observation[id_][1].shape)],traf,id_,route_keeper,type_)
                    del last_observation[id_]


    collisions_list.append(col_step)

    # if counter == HORIZON:
    #     reset()
    #     return
    #
    # if ac_counter == max_ac and num_ac == 0:
    #     reset()
    #     return



    # if num_ac == 0 and ac_counter != max_ac:
    #     print("@@@@@@@@@@@@ in return")
    #     return


    if not len(traf.id) == 0:
        ids = []
        new_actions = {}
        n_ac = len(traf.id)
        state = np.zeros((n_ac,5))

        id_sub = np.array(traf.id)[store_terminal != 1]
        ind = np.array([int(x[2:]) for x in traf.id])
        route = route_keeper[ind]

        state[:,0] = traf.lat
        state[:,1] = traf.lon
        state[:,2] = traf.tas
        state[:,3] = route
        state[:,4] = traf.ax

        norm_state,norm_context = getClosestAC(state,traf,route_keeper,previous_action,n_states,store_terminal,agent,last_observation,observation)


        policy = agent.act(norm_state,norm_context)


        # set_trace()
        # Max Policy

        # tmp = np.zeros((policy.shape[0], policy.shape[1]))
        # tmp[:, policy.shape[1]-1] = 1
        # policy = tmp




        for j in range(len(id_sub)):


            id_ = id_sub[j]

            # This is for updating s, sp, ...
            if not id_ in last_observation.keys():
                last_observation[id_] = [norm_state[j],norm_context[j]]

            if not id_ in observation.keys() and id_ in previous_action.keys():
                observation[id_] = [norm_state[j],norm_context[j]]

                agent.store(last_observation[id_],previous_action[id_],observation[id_],traf,id_,route_keeper)
                last_observation[id_] = observation[id_]

                del observation[id_]


            action = np.random.choice(agent.action_size,1,p=policy[j].flatten())[0]
            speed = agent.speeds[action]
            index = traf.id2idx(id_)

            # if action == 1: #hold
            #     speed = int(np.round((traf.cas[index]/tools.geo.nm)*3600))

            stack.stack('{} SPD {}'.format(id_,speed))
            new_actions[id_] = action


        total_reward += agent.rt

        lg.writeln("\n   t : " + str(counter) + " | rt : " + str(round(agent.rt, 3)))
        # print("Counter: ", counter, traf.ntraf, traf.id)
        # print("Counter: ", counter, traf.ntraf)
        previous_action = new_actions

    # print("Counter: ", counter, traf.ntraf)
    # print("Counter: ", counter, traf.ntraf, traf.id)
    if counter == HORIZON:
        reset()
        return


    counter += 1

def reset():
    global best_reward
    global counter
    global num_ac
    global num_success
    global success
    global collisions
    global num_collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start
    global num_entry
    global goal_time
    global collisions_list
    global total_reward

    if (agent.episode_count+1) % 5 == 0:
        agent.train()

    end = time.time()

    # print("%%%%%%%%%%%%%% reset()")


    goals_made = success

    num_success_train.append(success)
    num_collisions_train.append(collisions)



    # route_queue = random.choices(choices,k=positions.shape[0])
    route_queue = random.choices(choices, k=num_entry)

    previous_action = {}
    route_keeper = np.zeros(max_ac,dtype=int)
    last_observation = {}
    observation = {}

    t_success = np.array(num_success_train)
    t_coll = np.array(num_collisions_train)
    np.save(pro_folder+"/log/"+dir_name+'/success_train_B.npy',t_success)
    np.save(pro_folder+"/log/"+dir_name+'/collisions_train_B.npy',t_coll)

    if agent.episode_count > 150:
        df = pd.DataFrame(t_success)
        if float(df.rolling(150,150).mean().max()) >= best_reward:
            agent.save(True,case_study='B', mpath=pro_folder+"/log/"+dir_name+"/model")
            best_reward = float(df.rolling(150,150).mean().max())

            best_reward = float(df.rolling(150,150).mean().max())


    agent.save(case_study='B', mpath=pro_folder+"/log/"+dir_name+"/model")

    avg_tr = round(np.mean(goal_time), 3)
    max_cnf = np.max(collisions_list)
    avg_cnf = np.mean(collisions_list)

    # print("Episode: {} | Reward: {} | Best Reward: {}".format(agent.episode_count,goals_made,best_reward))
    lg.writeln("-------------------------------")
    lg.writeln("Episode : {}".format(agent.episode_count))
    lg.writeln("   Total Reward : {}".format(round(total_reward, 3)))
    lg.writeln("   Avg. Conflicts : " + str(round(avg_cnf, 3)))
    lg.writeln("   Max  Conflicts : " + str(max_cnf))
    lg.writeln("   Total Conflicts : " + str(collisions))
    lg.writeln("   Goal Reached : " + str(goals_made))
    lg.writeln("   Avg. Travel Time : " + str(avg_tr))


    writer.add_scalar('Reward/Total Rewards', total_reward, agent.episode_count)
    writer.add_scalar('Metrics/GoalReached', goals_made, agent.episode_count)
    writer.add_scalar('Metrics/AvgTravelTime', avg_tr, agent.episode_count)
    writer.add_scalar('Metrics/AvgConflicts', avg_cnf, agent.episode_count)
    writer.add_scalar('Metrics/TotalConflicts', collisions, agent.episode_count)
    writer.add_scalar('Metrics/MaxConflicts', max_cnf, agent.episode_count)

    agent.episode_count += 1

    success = 0
    collisions = 0
    counter = 0
    num_ac = 0
    ac_counter = 0
    agent.rt = 0
    total_reward = 0
    collisions_list = []
    goals_made = 0

    if agent.episode_count == agent.numEpisodes:
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')

    start = time.time()

def getClosestAC(state,traf,route_keeper,new_action,n_states,store_terminal,agent,last_observation,observation):
    n_ac = traf.lat.shape[0]
    norm_state = np.zeros((len(store_terminal[store_terminal!=1]),5))
    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1,1)
    d = geo.latlondist_matrix(np.repeat(state[:,0],n_ac),np.repeat(state[:,1],n_ac),np.tile(state[:,0],n_ac),np.tile(state[:,1],n_ac)).reshape(n_ac,n_ac)
    argsort = np.array(np.argsort(d,axis=1))
    total_closest_states = []
    route_count = 0
    i = 0
    j = 0
    max_agents = 1
    count = 0

    # set_trace()
    for i in range(d.shape[0]):
        r = int(state[i][3])
        # lat,lon,glat,glon,h = agent.positions[r]
        if store_terminal[i] == 1:
            continue
        # ownship_obj = LineString([[state[i][1],state[i][0],31000],[glon,glat,31000]])
        norm_state[count,:] = agent.normalize_that(state[i],'state',id_=traf.id[i])
        closest_states = []
        count += 1
        route_count = 0
        intruder_count = 0
        for j in range(len(argsort[i])):
            index = int(argsort[i][j])
            if i == index:
                continue
            if store_terminal[index] == 1:
                continue
            route = int(state[index][3])
            if route == r and route_count == 2:
                continue
            if route == r:
                route_count += 1
            # lat,lon,glat,glon,h = agent.positions[route]
            # int_obj = LineString([[state[index,1],state[index,0],31000],[glon,glat,31000]])
            # if not ownship_obj.intersects(int_obj):
            #     continue
            if d[i,index] > 100:
                continue
            max_agents = max(max_agents,j)
            if len(closest_states) == 0:
                closest_states = np.array([traf.lat[index], traf.lon[index], traf.tas[index],route,traf.ax[index]])
                closest_states = agent.normalize_that(norm_state[count-1],'context',closest_states,state[i],id_=traf.id[index])
            else:
                adding = np.array([traf.lat[index], traf.lon[index], traf.tas[index],route,traf.ax[index]])
                adding = agent.normalize_that(norm_state[count-1],'context',adding,state[i],id_=traf.id[index])
                closest_states = np.append(closest_states,adding,axis=1)
            intruder_count += 1
            if intruder_count == agent.num_intruders:
                break
        if len(closest_states) == 0:
            closest_states = np.array([0,0,0,0,0,0,0]).reshape(1,1,7)
        if len(total_closest_states) == 0:
            total_closest_states = closest_states
        else:
            total_closest_states = np.append(tf.keras.preprocessing.sequence.pad_sequences(total_closest_states,agent.num_intruders,dtype='float32'),tf.keras.preprocessing.sequence.pad_sequences(closest_states,agent.num_intruders,dtype='float32'),axis=0)
    if len(total_closest_states) == 0:
        # total_closest_states = np.array([0,0,0,0,0,0,0]).reshape(1,agent.num_intruders,7)
        # James
        total_closest_states = np.zeros((1, agent.num_intruders, 7), dtype='int64')
    return norm_state,total_closest_states

# ---------- James
def init():

    global dir_name, pro_folder

    if os.path.exists(pro_folder) is False:
        print("=============================================")
        print("Error: project folder path does not exist")
        print("=============================================")
        exit()


    os.system("mkdir "+pro_folder+"/log/")
    os.system("mkdir "+pro_folder+"/log/" + dir_name)
    deleteDir(pro_folder+"/log/"+dir_name + "/plots/")

    os.system("cp "+pro_folder+"/"+"parameters.py "+pro_folder+"/log/"+dir_name+"/")
    os.system("mkdir "+pro_folder+"/log/"+dir_name+"/plots")

    os.system("mkdir "+pro_folder+"/log/" + dir_name + "/model")

def add_wind():
    sp = np.random.randint(0, MAX_WIND)
    stack.stack('WIND 52.29145730495272, 4.777671441950008,2000,90,' + str(sp))

def add_poly():

    # Poly
    poly = np.array(np.load("./routes_james/"+MAP_ID+"/"+MAP_ID+"_poly.npy", allow_pickle=True))

    for p in range(poly.shape[0]):
        t1 = reduce(lambda x1, x2: " " + str(x1) + "," + str(x2), poly[p])
        stack.stack('POLY ' + str(p) + t1)
    routes_list = areas.keys()
    return routes_list
