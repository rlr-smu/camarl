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
import gym
import time
from gym.envs.registration import register

import os
from pprint import pprint
import time
from ipdb import set_trace
import pdb
import rlcompleter
from environments.grid_navigation.grid_navigation import CGMGridMoving as CGMgrid
from parameters import TOTAL_AGENTS, GRID, CAP
import numpy as np

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #

# def init_environment(env_name=None):


class environment:

    def __init__(self, env_name=None):

        self.env_name = env_name

        if self.env_name == 'grid_nav':

            self.env_main = CGMgrid(TOTAL_AGENTS, GRID, GRID, CAP)
            self.total_agents = TOTAL_AGENTS

            # ----- State space
            self.num_states = self.env_main.stateNum
            self.state_dim = 1

            # ----- Action space
            self.num_actions = self.env_main.actionNum
            self.action_dim = 1

            # ----- at_rew
            self.atomic_reward = None

            # ----- Nbr
            self.obs_dim = -1
            for k in self.env_main.neighbors:
                if self.obs_dim < len(self.env_main.neighbors[k]):
                    self.obs_dim = len(self.env_main.neighbors[k])
            self.obs_nbr = np.zeros((len(self.env_main.neighbors), self.obs_dim), dtype=np.int32)
            self.obs_nbr.fill(-1)
            for i in self.env_main.neighbors:
                nbr = self.env_main.neighbors[i]
                nbr.sort()
                self.obs_nbr[i][0:len(nbr)] = nbr

    def step(self, action_prob=None, action=None):

        nts, ntsa, ntsas, obs, action, global_reward, at_reward, dones, info = None, None, None, None, None, None, None, None, None
        if self.env_name == 'grid_nav':
            obs, action, global_reward, dones, info = self.env_main.step(action_prob)
            nts = info['state']
            ntsas = info['count']
            ntsa = info['count'].sum(-1)
            at_reward = info['rewards']

        return nts, ntsa, ntsas, action, global_reward, np.array([at_reward]), dones, info

    def reset(self):

        nts = None
        if self.env_name == 'grid_nav':
            obs, state_count = self.env_main.reset()
            nts = state_count.reshape(self.num_states, self.num_actions).sum(-1)

        return nts

class environment_old:

    def __init__(self, env_name=None):
        self.env_name = env_name
        self.env = None

    def init_env(self):

        if "multigrid" in self.env_name:
            if "oneroom" in self.env_name:
                register(
                    id='multigrid-oneroom-v0',
                    entry_point='gym_multigrid.envs:OneRoomEnvNxN',
                )
                self.env = gym.make('multigrid-oneroom-v0')

            elif "tworoom" in self.env_name:
                register(
                    id='multigrid-tworoom-v0',
                    entry_point='gym_multigrid.envs:TwoRoomEnv10x10',
                )
                self.env = gym.make('multigrid-tworoom')
            elif "threeroom" in self.env_name:
                register(
                    id='multigrid-threeroom-v0',
                    entry_point='gym_multigrid.envs:ThreeRoomEnvNxN',
                )
                self.env = gym.make('multigrid-threeroom-v0')
            elif "fourroom" in self.env_name:
                register(
                    id='multigrid-fourroom-v0',
                    entry_point='gym_multigrid.envs:FourRoomEnvNxN',
                )
                self.env = gym.make('multigrid-fourroom-v0')
            # set_trace()
            self.action_space = self.env.actions.available
            self.total_agents = len(self.env.agents)

        elif "grid_nav" in self.env_name:
            self.env = CGMgrid(TOTAL_AGENTS, GRID, GRID)

            self.env.total_agents = TOTAL_AGENTS

            # ----- State space
            self.env.num_states = self.env.stateNum
            self.env.state_dim = 1

            # ----- Action space
            self.env.num_actions = self.env.actionNum
            self.env.action_dim = 1

            # ----- at_rew
            self.env.atomic_reward = None

            # ----- Nbr
            self.env.obs_dim = -1
            for k in self.env.neighbors:
                if self.env.obs_dim < len(self.env.neighbors[k]):
                    self.env.obs_dim = len(self.env.neighbors[k])
            self.env.obs_nbr = np.zeros((len(self.env.neighbors), self.env.obs_dim), dtype=np.int32)
            self.env.obs_nbr.fill(-1)
            for i in self.env.neighbors:
                nbr = self.env.neighbors[i]
                nbr.sort()
                self.env.obs_nbr[i][0:len(nbr)] = nbr

        else:
            print("No Environment Exists")
            exit()

        return self.env

def main():
    print("Hello World")

# =============================================================================== #

if __name__ == '__main__':
    main()
