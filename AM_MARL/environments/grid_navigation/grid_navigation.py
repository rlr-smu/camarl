import numpy as np
from os.path import dirname, abspath
import os
import sys
sys.path.append(os.getcwd()+"/environments/grid_navigation")
from spaces.product import Product
import gym
from spaces.box import Box
from special import probScale
from ipdb import set_trace
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from matplotlib.patches import Rectangle
from pprint import pprint


class vizualize:

    def __init__(self, grid=None):

        self.grid = grid
        self.getGridInfo()

    def getGridInfo(self):

        h, w = 1, 1
        self.grid_info = {}

        gid = 0
        for x in range(self.grid):
            for y in range(self.grid):
                self.grid_info[gid] = [x, h, y, w]
                gid += 1


        if self.grid == 5:
            self.goal = [20]
            self.obstacle = [10, 11, 13, 14]

    def genRandomXY(self, xc, yc, count):

        x_list = []
        y_list = []
        eps = 0.2
        for c in range(count):
            x_list.append(np.random.uniform(xc[0]+eps, xc[1]-eps))
            y_list.append(np.random.uniform(yc[0]+eps, yc[1]-eps))
        return x_list, y_list

    def plot_rect(self, xs, ys, w, z):

        x_list = [xs, xs+w, xs+w, xs, xs]
        y_list = [ys, ys, ys+w, ys+w, ys]

        plt.plot(x_list, y_list, linewidth=0.4, color='black')

        if z in self.obstacle:
            plt.fill_between([xs, xs+w], [ys, ys], [ys+w, ys+w], color='grey')

        if z in self.goal:
            plt.fill_between([xs, xs+w], [ys, ys], [ys+w, ys+w], color='green', alpha=0.9)

    def topology(self):

        for z in self.grid_info.keys():
            xs = self.grid_info[z][0]
            ys = self.grid_info[z][2]
            self.plot_rect(xs, ys, 1, z)

    def step(self, states):

        plt.ion()
        plt.clf()
        plt.xlim(-1, self.grid+1)
        plt.ylim(-1, self.grid+1)
        plt.axis('off')
        plt.grid(False)
        self.topology()
        for z in self.grid_info.keys():

            # print(z, self.grid_info[z])

            xc = (self.grid_info[z][0], self.grid_info[z][1])
            yc = (self.grid_info[z][2], self.grid_info[z][3])
            x, y = self.genRandomXY(xc, yc, states[z])
            plt.scatter(x, y, color='red', s=5, marker='s')

        plt.draw()
        plt.pause(0.1)

class CGMGridMoving(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, N, edge1 = 5, edge2 = 5, CAP=None):
        name = 'FictitiousCongestionPW'
        d = dirname(dirname(abspath(__file__)))
        self.stateNum = edge1*edge2
        self.actionNum = 5#int(inputFile.readline())
        self.N = N
        self.remaining = 0
        # self.N = 1
        self.pwNum = 5
        initialDistribution = np.zeros(self.stateNum)
        initialDistribution[0:edge1] = 0.1
        stateIndex = np.array(range(self.stateNum)).reshape(edge1, edge2)
        actionIndex = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
        transitedState = np.tile(range(self.stateNum), (self.actionNum, 1))
        for a in range(1, self.actionNum):
            for s1 in range(edge1):
                for s2 in range(edge2):
                    if ((s1 + actionIndex[a, 0]) in range(edge1)) and ((s2 + actionIndex[a, 1]) in range((edge2))):
                        transitedState[a, stateIndex[s1, s2]] = stateIndex[
                            s1 + actionIndex[a, 0], s2 + actionIndex[a, 1]]

        self.P = np.zeros((self.pwNum, self.actionNum, self.stateNum, self.stateNum), float)
        capacity = np.ones((self.stateNum, self.actionNum)) * CAP  # np.random.randint(4, size=(stateNum, actionNum)) + 1

        successProb = np.random.random((self.stateNum, self.actionNum))
        for s in range(0, self.stateNum - 1):
            for a in range(0, self.actionNum):
                for p in range(self.pwNum):
                    if transitedState[a, s] != s:
                        if (p + 1) <= capacity[s, a]:
                            self.P[p, a, s, transitedState[a, s]] = 0.8  # successProb[s, a]#np.exp(-1 * (ud[p] - 1))#0.8#
                        else:
                            self.P[p, a, s, transitedState[a, s]] = 0.8 * capacity[s, a] / (p + 1)
                        self.P[p, a, s, s] = 1 - self.P[p, a, s, transitedState[a, s]]
                    else:
                        self.P[p, a, s, s] = 1

        self.P[:, :, self.stateNum - 1, 0:edge1] = 1.0 / float(edge1)

        self.R = np.zeros((self.pwNum, self.stateNum, self.actionNum))  # np.random.random((H, pwNum, stateNum, actionNum))#
        for s in range(0, self.stateNum):
            for a in range(0, self.actionNum):
                for p in range(self.pwNum):
                    if (p + 1) > capacity[s, a]:
                        self.R[p, s, a] = -1

        self.R[:, self.stateNum - 1, 0] = 1


        self.initialDistribution1 = np.append(initialDistribution, [1 - np.sum(initialDistribution)])
        self.S = np.random.multinomial(self.N, self.initialDistribution1)[0:self.stateNum]
        self.low = 0
        self.high = self.N
        self.state = self.S
        tempA = np.zeros((self.stateNum, 2)).astype(int)
        tempA[:, 1] = self.actionNum - 1
        self.shape = (self.stateNum, self.actionNum)
        self.goalState = self.stateNum - 1
        self.edge1 = int(np.sqrt(self.stateNum))
        self.edge2 = int(np.sqrt(self.stateNum))
        stateIndex = np.array(range(self.stateNum)).reshape(self.edge1, self.edge2)
        actionIndex = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
        transitedState = np.tile(range(self.stateNum), (self.actionNum, 1))
        self.neighbors = {}
        for s1 in range(self.edge1):
            for s2 in range(self.edge2):
                s = stateIndex[s1, s2]
                tempNeighbor = []
                for a in range(1, self.actionNum):
                    if ((s1 + actionIndex[a, 0]) in range(self.edge1)) and (
                        (s2 + actionIndex[a, 1]) in range((self.edge2))):
                        tempNeighbor.append(stateIndex[s1 + actionIndex[a, 0], s2 + actionIndex[a, 1]])
                self.neighbors.update({s: tempNeighbor})

    def reset(self):
        self.S = np.random.multinomial(self.N, self.initialDistribution1)[0:self.stateNum]
        self.state = self.S
        self.remaining = self.N - np.sum(self.S)
        obs = []
        for i in range(self.stateNum):
            temp = np.zeros((5))
            temp[0] = self.S[i]
            tempIndex = 1
            for j in self.neighbors.get(i):
                temp[tempIndex] = self.S[j]
                tempIndex += 1
            temp /= float(self.N)
            obs.append(temp.flatten())
        self.state_count = np.tile(self.state[:, np.newaxis], [1, self.actionNum]).flatten()

        return obs, self.state_count

    @property
    def _state(self):
        return self.state

    @property
    def observation_space(self):
        components = []
        for i in range(self.stateNum):
            components.append(Box(low=self.low, high=self.high, shape=5))#self.stateNum
        return Product(components)

    @property
    def action_space(self):
        components = []
        for i in range(self.stateNum):
            components.append(Box(low=0, high=self.high, shape=(self.actionNum)))
        return Product(components)

    def step(self, prob):
        prob = probScale(np.asarray(prob, dtype=float))
        a = np.asarray(
            [np.random.multinomial(self.state[i], prob[i]) for i in range(prob.shape[0])])
        tempA = np.copy(a)
        tempA = np.maximum(tempA - 1, 0)
        tempA[tempA >= self.pwNum] = self.pwNum - 1
        SAS = np.asarray(
            [[np.random.multinomial(a[i, j], self.P[tempA[i, j], j, i]) for j in range(a.shape[1])] for i in
             range(a.shape[0])])
        # reward = np.zeros((self.stateNum, self.actionNum))
        # reward = reward  # /np.maximum(a, 0.1)
        # reward[self.goalState, 0] = a[self.goalState, 0]*10
        self.S = np.sum(SAS, axis=(0, 1)).astype(int)
        rewards = np.asarray(
            [[self.R[tempA[i, j], i, j] for j in range(a.shape[1])] for i in range(a.shape[0])])*(a>0)
        reward = np.sum(rewards * a)

        if self.remaining > 0:
            self.S += np.random.multinomial(self.remaining, self.initialDistribution1)[0:self.stateNum]
            self.remaining = self.N - np.sum(self.S)
        self.state = self.S


        obs = []
        for i in range(self.stateNum):
            temp = np.zeros((5))
            temp[0] = self.S[i]
            tempIndex = 1
            for j in self.neighbors.get(i):
                temp[tempIndex] = self.S[j]
                tempIndex += 1
            temp /= float(self.N)
            obs.append(temp.flatten())
        return obs,  a, reward, False, {'state': self.state, 'count': SAS, 'rewards': rewards}# self.state_count,

        # return Step(observation=obs, reward=reward, trueReward=reward, done=False, count = SAS)

    def _close(self):
        self.state = None

    def _render(self, mode="human", close=False):
        pass
