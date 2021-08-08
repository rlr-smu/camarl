import os
import numpy as np
import random
import time
from copy import copy
from collections import deque
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
import geopy.distance
from bluesky.tools import geo
from bluesky import traf
from operator import itemgetter
from shapely.geometry import LineString
import numba as nb
from ipdb import set_trace
from parameters import LEARNING_RATE, CASE, SEED, ALT, REAL_START_TIME, HORIZON
import torch as tc
from torch.distributions import Categorical
################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################
from torch.utils.tensorboard import SummaryWriter
from bluesky.tools.areafilter import checkInside, areas

LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-4
HIDDEN_SIZE = 32
import time

@nb.njit()
def discount(r,discounted_r,cumul_r):
    """ Compute the gamma-discounted rewards over an episode
    """
    for t in range(len(r)-1,-1,-1):
        cumul_r = r[t] + cumul_r * 0.99
        discounted_r[t] = cumul_r
    return discounted_r

def dist_goal(states,traf,i):
    olat,olon=states
    # ilat,ilon =traf.ap.route[i].wplat[0],traf.ap.route[i].wplon[0]
    # James
    # print("-------------")
    # print(i)
    # print(traf.id[i])
    # print(traf.ap.route[i].wplat)
    # print(traf.ap.route[i].wplon)

    if len(traf.ap.route[i].wplat)  == 0:
        return 0

    ilat, ilon = traf.ap.route[i].wplat[-1], traf.ap.route[i].wplon[-1]
    dist = geo.latlondist(olat,olon,ilat,ilon)/geo.nm
    return dist

def getClosestAC_Distance(self, state,traf,route_keeper):

    olat,olon,ID = state[:3]
    index = int(ID[2:])
    rte = int(route_keeper[index])

    # lat,lon,glat,glon,h = self.positions[rte]
    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1,1)
    # ownship_obj = LineString([[olon,olat,31000],[glon,glat,31000]])
    d  = geo.latlondist_matrix(np.repeat(olat,size),np.repeat(olon,size),traf.lat,traf.lon)
    d = d.reshape(-1,1)
    dist = np.concatenate([d,index],axis=1)
    dist = sorted(np.array(dist),key=itemgetter(0))[1:]
    if len(dist) > 0:
        for i in range(len(dist)):
            # index = int(dist[i][1])
            # ID_ = traf.id[index]
            # index_route = int(ID_[2:])
            # rte_int = route_keeper[index_route]
            # lat,lon,glat,glon,h = self.positions[rte_int]
            # int_obj = LineString([[traf.lon[index],traf.lat[index],31000],[glon,glat,31000]])
            # if not ownship_obj.intersects(int_obj):
            #     continue
            # print("$$$",self.intersection_distances, rte)
            # if len(self.intersection_distances) > 0 : # By James
            #     if not rte_int in self.intersection_distances[rte].keys() and rte_int != rte:
            #         continue
            if dist[i][0] > 100:
                continue
            return dist[i][0]
    else:
        return np.inf
    return np.inf

def getClosestAC(state,traf,store_terminal,agent):
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

            if len(agent.intersection_distances) > 0: #By James
                if not route in agent.intersection_distances[r].keys() and route != r:
                    continue


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
        # set_trace()
        # total_closest_states = np.array([0,0,0,0,0,0,0]).reshape(1,agent.num_intruders,7)
        total_closest_states = np.array([[[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0]]])



    return norm_state,total_closest_states

def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):

        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))

    return loss

# initalize the PPO agent
class PPO_Agent:
    def __init__(self,state_size,action_size,num_routes,numEpisodes,positions,num_intruders, load_model=False, dir_name=None, actions=None, routes_list=None):

        tf.random.set_random_seed(SEED)
        np.random.seed(SEED)

        self.pro_folder = os.getcwd()
        self.state_size = state_size
        self.action_size = action_size
        self.positions = positions
        # self.positions_out = positions_out
        self.gamma = 0.99    # discount rate
        self.numEpisodes = numEpisodes
        self.max_time = 500
        self.num_intruders = num_intruders
        self.routes_list = routes_list

        self.episode_count = 0
        # self.speeds = np.array([156,0,346])
        # self.speeds = np.array([100, 200, 300, 400, 500])
        self.speeds = actions
        self.max_agents = 0
        self.num_routes = num_routes
        self.experience = {}
        self.dist_close = {}
        self.dist_goal = {}
        self.tas_max = 253.39054470774
        self.tas_min = 118.54804803287088
        # self.lr = 0.0001
        self.lr = LEARNING_RATE
        self.value_size = 1

        self.getRouteDistances()

        self.model_check = []
        self.model = self._build_PPO()

        if load_model:
            # self.model.load_weights = self.load(self.pro_folder+"/load_model/model_"+CASE+"_ppo.h5")
            if os.path.exists(self.pro_folder+"/load_model/best_model_"+CASE+"_ppo.h5"):
                self.model.load_weights = self.load(self.pro_folder+"/load_model/best_model_"+CASE+"_ppo.h5")
            else:
                self.model.load_weights = self.load(self.pro_folder+"/load_model/model_"+CASE+"_ppo.h5")


        self.count = 0
        self.rt = 0
        self.dir_name = dir_name
        self.dist_close_james = {}
        self.writer = SummaryWriter(self.pro_folder+"/log/"+self.dir_name+"/plots/")

    def getRouteDistances(self):
        self.intersections = {}
        self.intersection_distances = {}
        self.route_distances = []
        self.conflict_routes = {}

        for i in range(len(self.positions)):

            wps = self.positions[i][:-2]
            i = 0
            dist = 0
            while i < len(wps) - 2:
                lat1 = wps[i]
                lon1 = wps[i+1]
                lat2 = wps[i+2]
                lon2 = wps[i+3]
                i += 2
                _, d = geo.qdrdist(lat1,lon1,lat2,lon2)
                dist += d
            self.route_distances.append(dist)

        self.max_d = max(self.route_distances)

    def normalize_that(self,value,what,context=False,state=False,id_=None):

        if what=='spd':

            if value > self.tas_max:
                self.tas_max = value

            if value < self.tas_min:
                self.tas_min = value
            return (value-self.tas_min)/(self.tas_max-self.tas_min)

        if what=='rt':
            # By James
            if self.num_routes > 1:
                return value / (self.num_routes - 1)
            else:
                return value

        if what=='state':

            dgoal = self.dist_goal[id_]/self.max_d
            spd = self.normalize_that(value[2],'spd')
            rt = self.normalize_that(value[3],'rt')
            acc = value[4]+0.5
            rt_own = int(value[3])
            norm_array = np.array([dgoal,spd,rt,acc,3/self.max_d])
            return norm_array

        if what == 'context':

            rt_own = int(state[3])
            dgoal = self.dist_goal[id_]/self.max_d
            spd = self.normalize_that(context[2],'spd')
            rt = self.normalize_that(context[3],'rt')
            acc = context[4]+0.5
            rt_int = int(context[3])
            dist_own_intersection = 0
            dist_int_intersection = 0

            if rt_own == rt_int:
                dist_away = abs(value[0]-dgoal)
                dist_own_intersection = 0
                dist_int_intersection = 0#
            else:
                # dist_own_intersection = abs(self.intersection_distances[rt_own][rt_int]/self.max_d - value[0])
                # dist_int_intersection = abs(self.intersection_distances[rt_int][rt_own]/self.max_d - dgoal)
                d  = geo.latlondist(state[0],state[1],context[0],context[1])/geo.nm
                dist_away = d/self.max_d


            context_arr = np.array([dgoal,spd,rt,acc,dist_away,dist_own_intersection,dist_int_intersection])

            return context_arr.reshape(1,1,7)

    def _build_PPO(self):

        I = tf.keras.layers.Input(shape=(self.state_size,),name='states')

        context = tf.keras.layers.Input(shape=(self.num_intruders,7),name='context')
        empty = tf.keras.layers.Input(shape=(HIDDEN_SIZE,),name='empty')

        advantage = tf.keras.layers.Input(shape=(1,),name=CASE)
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,),name='old_pred')

        flatten_context = tf.keras.layers.Flatten()(context)
        # encoding other_state into 32 values
        H1_int = tf.keras.layers.Dense(HIDDEN_SIZE,activation='relu')(flatten_context)
        # now combine them
        combined = tf.keras.layers.concatenate([I,H1_int], axis=1)


        H2 = tf.keras.layers.Dense(256,activation='relu')(combined)
        H3 = tf.keras.layers.Dense(256,activation='relu')(H2)

        output = tf.keras.layers.Dense(self.action_size+1,activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:,:self.action_size],output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:,self.action_size:],output_shape=(self.value_size,))(output)

        # now I need to apply activation
        policy_out = tf.keras.layers.Activation('softmax',name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear',name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)

        model = tf.keras.models.Model(inputs=[I,context,empty,advantage,old_prediction], outputs=[policy_out,value_out])

        self.predictor = tf.keras.models.Model(inputs=[I,context,empty], outputs=[policy_out,value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out':proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction), 'value_out':'mse'})

        # print(model.summary())

        return model

    # By James
    def paper_reward(self, id_):

        reward = 0
        dist = self.dist_close_james[id_]

        if dist < 3:
            reward = -1


        if dist < 10 and dist >= 3:
            reward = -0.1 + 0.05 * (dist)

        if dist >= 10:
            reward = 0

        self.rt += reward

    def store(self,state,action,next_state,traf,id_,route_keeper,term=0):
        reward = 0
        done = False


        # James
        dist = self.dist_close[id_]
        if dist < 3:
            reward= -1
            done = True

        elif dist < 10 and dist >= 3:
            reward = -0.1 + 0.05 * (dist)


        elif dist >= 10:
            reward = 0
            done = True


        '''                
        if term == 0:
            lat = traf.lat[traf.id2idx(id_)]
            lon = traf.lon[traf.id2idx(id_)]
            dist = self.dist_close[id_]
            if dist < 10 and dist >= 3:
                # reward = -0.1 + 0.05*(dist/10)
                reward = -0.1 + 0.05 * (dist)
        if term == 1:
            reward= -1
            done = True
        if term == 2:
            reward = 0
            done = True
        '''

        self.rt += reward

        state,context = state
        state = state.reshape((1,5))
        context = context.reshape((1,-1,7))

        if context.shape[1] > self.num_intruders:
            context = context[:,-self.num_intruders:,:]

        self.max_agents = max(self.max_agents,context.shape[1])

        if not id_ in self.experience.keys():
            self.experience[id_] = {}

        try:
            self.experience[id_]['state'] = np.append(self.experience[id_]['state'],state,axis=0)

            if self.max_agents > self.experience[id_]['context'].shape[1]:
                self.experience[id_]['context'] = np.append(tf.keras.preprocessing.sequence.pad_sequences(self.experience[id_]['context'],self.max_agents,dtype='float32'),context,axis=0)
            else:
                self.experience[id_]['context'] = np.append(self.experience[id_]['context'],tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32'),axis=0)

            self.experience[id_]['action'] = np.append(self.experience[id_]['action'],action)
            self.experience[id_]['reward'] = np.append(self.experience[id_]['reward'],reward)
            self.experience[id_]['done'] = np.append(self.experience[id_]['done'],done)


        except:
            self.experience[id_]['state'] = state
            if self.max_agents > context.shape[1]:
                self.experience[id_]['context'] = tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32')
            else:
                self.experience[id_]['context'] = context

            self.experience[id_]['action'] = [action]
            self.experience[id_]['reward'] = [reward]
            self.experience[id_]['done'] = [done]

    def train(self, ep=None):

        """Grab samples from batch to train the network"""

        total_state = []
        total_reward = []
        total_A = []
        total_advantage = []
        total_context = []
        total_policy = []

        total_length = 0

        for transitions in self.experience.values():
            episode_length = transitions['state'].shape[0]
            total_length += episode_length

            state = transitions['state']#.reshape((episode_length,self.state_size))
            context = transitions['context']
            reward = transitions['reward']
            done = transitions['done']
            action  = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount(reward,discounted_r, cumul_r)
            policy,values = self.predictor.predict({'states':state,'context':context,'empty':np.zeros((len(state),HIDDEN_SIZE))},batch_size=256)
            advantages = np.zeros((episode_length, self.action_size))
            index = np.arange(episode_length)
            advantages[index,action] = 1
            A = discounted_rewards - values[:,0]

            if len(total_state) == 0:

                total_state = state
                if context.shape[1] == self.max_agents:
                    total_context = context
                else:
                    total_context = tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32')
                total_reward = discounted_rewards
                total_A = A
                total_advantage = advantages
                total_policy = policy

            else:
                total_state = np.append(total_state,state,axis=0)
                if context.shape[1] == self.max_agents:
                    total_context = np.append(total_context,context,axis=0)
                else:
                    total_context = np.append(total_context,tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32'),axis=0)
                total_reward = np.append(total_reward,discounted_rewards,axis=0)
                total_A = np.append(total_A,A,axis=0)
                total_advantage = np.append(total_advantage,advantages,axis=0)
                total_policy = np.append(total_policy,policy,axis=0)


        total_A = (total_A - total_A.mean())/(total_A.std() + 1e-8)
        self.model.fit({'states':total_state,'context':total_context,'empty':np.zeros((total_length,HIDDEN_SIZE)),CASE:total_A,'old_pred':total_policy}, {'policy_out':total_advantage,'value_out':total_reward}, shuffle=True,batch_size=total_state.shape[0],epochs=8, verbose=0)


        self.max_agents = 0
        self.experience = {}

    def load(self, name):
        print('Loading weights...')
        self.model.load_weights(name)
        print('Successfully loaded model weights from {}'.format(name))

    def save(self,best=False,case_study=CASE, mpath=None):


        if best:

            self.model.save_weights(mpath+"/"+'best_model_{}_ppo.h5'.format(case_study))

        else:

            self.model.save_weights(mpath+"/"+'model_{}_ppo.h5'.format(case_study))

    # action implementation for the agent
    def act(self,state,context):

        if state.shape[0] == 0:
            # James
            policy = np.zeros((traf.ntraf, 3))
            policy[:,1] = 1
        else:
            context = context.reshape((state.shape[0],-1,7))
            if context.shape[1] > self.num_intruders:
                context = context[:,-self.num_intruders:,:]
            if context.shape[1] < self.num_intruders:
                context = tf.keras.preprocessing.sequence.pad_sequences(context,self.num_intruders,dtype='float32')
            policy,value = self.predictor.predict({'states':state,'context':context,'empty':np.zeros((state.shape[0],HIDDEN_SIZE))},batch_size=state.shape[0])
        return policy

    # By James
    def get_action(self,state,context):

        context = context.reshape((state.shape[0],-1,7))
        if context.shape[1] > self.num_intruders:
            context = context[:,-self.num_intruders:,:]
        if context.shape[1] < self.num_intruders:
            context = tf.keras.preprocessing.sequence.pad_sequences(context,self.num_intruders,dtype='float32')

        policy,value = self.predictor.predict({'states':state,'context':context,'empty':np.zeros((state.shape[0],HIDDEN_SIZE))},batch_size=state.shape[0])


        action_ac = np.zeros(traf.ntraf, dtype=int)
        for j in range(traf.ntraf):
            action = np.random.choice(self.action_size,1,p=policy[j].flatten())[0]
            action_ac[j] = action
        return action_ac

    # By James
    def store_rollouts(self, buff_nt=None, buff_ntellv=None, buff_rt=None, buff_ntev_mean=None,buff_act_prob=None):
        pass

    # By James
    def log(self, ep, ep_rw, buff_act_prob, avg_tr, avg_cnf, tot_cnf, goal_reached, mean_count):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)

    # By James
    def save_model(self, tot_reward, ep):
        pass

    # By James
    def clear_buffer(self):
        pass

    def update_james(self,traf,index,route_keeper):

        dist = getClosestAC_Distance(self, [traf.lat[index], traf.lon[index], traf.id[index]], traf, route_keeper)
        self.dist_close_james[traf.id[index]] = dist

    def update(self,traf,index,route_keeper):
        """calulate reward and determine if terminal or not"""
        T = 0
        type_ = 0
        dist = getClosestAC_Distance(self, [traf.lat[index], traf.lon[index], traf.id[index]], traf, route_keeper)
        if dist < 3:
            T = True
            type_ = 1
        self.dist_close[traf.id[index]] = dist

        d_goal = dist_goal([traf.lat[index], traf.lon[index]], traf, index)


        # x, y = traf.lat[index], traf.lon[index]

        # e = int(self.get_edges(x, y, ALT))
        # idx = traf.id[index]
        # if e == -1:
        #     T = True
        #     type_ = 2
        if d_goal < 5 and T == 0:
            T = True
            type_ = 2
        self.dist_goal[traf.id[index]] = d_goal
        return T,type_

    def get_edges(self, x, y, alt):

        for r in self.routes_list:
            if checkInside(r, x, y, alt):
                return r
        return -1
