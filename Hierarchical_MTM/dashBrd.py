"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 15 Apr 2019
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
import auxLib as ax
import pdb
import rlcompleter
from auxLib import tsplot

# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()
from data import env_data
import numpy as np
from parameters import NUM_OPTIONS, HORIZON, MAP_ID,  wDelay, wResource,capRatio, TOTAL_VESSEL
import matplotlib.style as style
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
style.use("seaborn")
import numpy as np
import networkx as nx
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, help='dashboard to run')
parser.add_argument('-a', '--agent', type=str, help='Agent to run')
args = parser.parse_args()

# =============================== Variables ================================== #


# ============================================================================ #

dt = env_data(mapName=MAP_ID)
planningZONES = dt.planningZONES
zGraph = dt.zGraph


class drawOption:

    def __init__(self, path="", save_counter=5):

        self.counter = 0
        self.save_counter = save_counter
        self.fcount = -1
        self.path = path
        self.op_color = ['red', 'green', 'blue']
        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                self.fcount += 2
        self.config()
        self.op_show2()
        # self.run()

    def config(self):
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.055, wspace=0.14, hspace=0.44)

        self.totRow = int((self.fcount + 1)/2)
        self.totCol = 2
        self.ax = []

        i = 1
        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                self.ax.append(plt.subplot(self.totRow, self.totCol, i))
                self.ax[i - 1].set_yticks(np.arange(0, NUM_OPTIONS, step=1))

                self.ax[i - 1].set_xlabel("Time")
                self.ax[i - 1].set_ylabel("Options")
                self.ax[i - 1].set_title("Zones: "+str(z) +"_"+str(zp)+" | Option Selected")


                self.ax.append(plt.subplot(self.totRow, self.totCol, i+1))
                self.ax[i].set_xlabel("Time")
                self.ax[i].set_ylabel("xi value")
                self.ax[i].set_title("Zones: "+str(z) +"_"+str(zp)+" | Option termination value")
                i += 2

    def update(self, _ ):

        self.counter += 1
        i = 1
        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                data = open(self.path +"op_"+str(z)+"_"+str(zp)+".txt", 'r').read()
                lines = data.split('\n')
                op = []
                xi = []
                for line in lines:
                    if len(line) > 1:
                        o,x = line.split(",")
                        op.append(float(o))
                        xi.append(float(x))

                op = op[-HORIZON:]
                xi = xi[-HORIZON:]

                for o in range(NUM_OPTIONS):
                    o_tmp = []
                    x = []
                    for t in range(len(op)):
                        if op[t] == o:
                            x.append(t)
                            o_tmp.append(o)
                    self.ax[i - 1].scatter(x, o_tmp, c=self.op_color[o], s=25)

                xi_all = {0:[], 1:[], 2:[]}
                for t in range(len(xi)):
                    o = op[t]
                    for j in range(NUM_OPTIONS):
                        if j == o:
                            xi_all[j].append(xi[t])

                for o in range(NUM_OPTIONS):
                    self.ax[i].plot(xi_all[o], color=self.op_color[o], label=str(o))
                i += 2

        if self.counter % self.save_counter == 0:
            plt.legend()
            # plt.savefig("log/"+dirName+"/opResult.png")
            plt.savefig(self.path + "/opResult.png")

    def run(self):

        self.ani = FuncAnimation(self.fig, self.update, interval=1000)
        plt.legend()
        plt.show()

    def op_show(self):

        i = 1
        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                data = open(self.path +"op_"+str(z)+"_"+str(zp)+".txt", 'r').read()
                lines = data.split('\n')
                op = []
                xi = []
                for line in lines:
                    if len(line) > 1:
                        o,x = line.split(",")
                        op.append(float(o))
                        xi.append(float(x))
                # self.ax[i-1].plot(op, color='black')
                self.ax[i - 1].scatter([ _ for _ in range(len(op))], op, c='black', s=5)

                xi_all = {0:[], 1:[], 2:[]}
                for t in range(len(xi)):
                    o = op[t]
                    for j in range(NUM_OPTIONS):
                        if j == o:
                            xi_all[j].append(xi[t])
                        else:
                            if len(xi_all[j]) == 0:
                                xi_all[j].append(0.5)
                            else:
                                last = xi_all[j][-1]
                                # pdb.set_trace()
                                xi_all[j].append(last)
                # pdb.set_trace()
                # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
                # pdb.set_trace()

                for o in range(NUM_OPTIONS):
                    self.ax[i].plot(xi_all[o], color=self.op_color[o], label=str(o))
                    # self.ax[i].legend()

                i += 2
        plt.legend()
        plt.show()

    def op_show2(self):

        i = 1
        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                data = open(self.path +"op_"+str(z)+"_"+str(zp)+".txt", 'r').read()
                lines = data.split('\n')
                op = []
                xi = []
                for line in lines:
                    if len(line) > 1:
                        o,x = line.split(",")
                        op.append(float(o))
                        xi.append(float(x))


                op = op[-HORIZON:]
                xi = xi[-HORIZON:]

                for o in range(NUM_OPTIONS):
                    o_tmp = []
                    x = []
                    for t in range(len(op)):
                        if op[t] == o:
                            x.append(t)
                            o_tmp.append(o)

                    self.ax[i - 1].scatter(x, o_tmp, c=self.op_color[o], s=25)

                xi_all = {0:[], 1:[], 2:[]}
                for t in range(len(xi)):
                    o = op[t]
                    for j in range(NUM_OPTIONS):
                        if j == o:
                            xi_all[j].append(xi[t])

                for o in range(NUM_OPTIONS):
                    self.ax[i].plot(xi_all[o], color=self.op_color[o], label=str(o))

                i += 2
        plt.show()

class dashboard:

    def __init__(self, data={}, save_counter=5):

        self.counter = 0
        self.save_counter = save_counter
        self.data = data
        self.path = data['path']
        self.num_plots = len(data) - 1
        self.config()
        self.run()

    def config(self):
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.055, wspace=0.14, hspace=0.44)
        self.ax = []

        sbplots_row = 3
        sbplots_col = 2

        for p in range(1, self.num_plots+1):
            self.ax.append(plt.subplot(sbplots_row, sbplots_col, p))
            self.ax[p-1].set_xlabel(self.data[p]['xlabel'])
            self.ax[p-1].set_ylabel(self.data[p]['ylabel'])
            self.ax[p-1].set_title(self.data[p]['title'])

    def update(self, _ ):

        self.counter += 1
        for p in range(1, self.num_plots+1):
            graph_data = open(self.data['path']+self.data[p]['fname'], 'r').read()
            lines = graph_data.split('\n')
            xs = []
            ys = []
            for line in lines:
                if len(line) > 1:
                    x, y = line.split(',')
                    xs.append(float(x))
                    ys.append(float(y))
            self.ax[p-1].plot(xs, ys, color='blue')

        if self.counter % self.save_counter == 0:
            plt.savefig(self.path+"/Result.png")

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=1000)
        plt.show()

class lgFile:

    def __init__(self, param=[], path=""):

        self.param = param
        self.path = path
        for p in self.param:
            f = open(self.path+p+".txt", 'w')
            f.close()

        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                f = open(self.path+"op_"+str(z)+"_"+str(zp)+".txt", 'w')
                f.close()

    def update(self, data={}):

        for p in self.param:
            with open(self.path+p+".txt", 'a') as f:
                f.writelines(str(data['x'])+","+str(data[p])+"\n")

    def updateOptionLog(self,  nt, o_all, xi_all):

        for z in planningZONES:
            zp_nbr = nx.neighbors(zGraph, z)
            for zp in zp_nbr:
                if nt[z] > 0 or nt[zp] > 0:
                    o = o_all[z][zp]
                    xi = xi_all[z][zp]
                    with open(self.path +"op_"+str(z)+"_"+str(zp)+".txt", 'a') as f:
                        f.writelines(str(o)+","+str(xi)+"\n")

def dash(path=""):

    data = {}
    data['path'] = path
    data[1] = {}
    data[1]['fname'] = "reward.txt"
    data[1]['xlabel'] = "Episodes"
    data[1]['ylabel'] = "Total Reward"
    data[1]['title'] = "Total Reward"

    data[2] = {}
    data[2]['fname'] = "q_loss.txt"
    data[2]['xlabel'] = "Episodes"
    data[2]['ylabel'] = "Q_Loss"
    data[2]['title'] = "Q_Loss"

    data[3] = {}
    data[3]['fname'] = "vio.txt"
    data[3]['xlabel'] = "Episodes"
    data[3]['ylabel'] = "Total Violation"
    data[3]['title'] = "Total Violation"

    data[4] = {}
    data[4]['fname'] = "delay.txt"
    data[4]['xlabel'] = "Episodes"
    data[4]['ylabel'] = "Total Delay"
    data[4]['title'] = "Total Delay"

    data[5] = {}
    data[5]['fname'] = "beta_loss.txt"
    data[5]['xlabel'] = "Episodes"
    data[5]['ylabel'] = "Beta_Loss"
    data[5]['title'] = "Beta_Loss"

    data[6] = {}
    data[6]['fname'] = "total_loss.txt"
    data[6]['xlabel'] = "Episodes"
    data[6]['ylabel'] = "total_Loss"
    data[6]['title'] = "total_Loss"


    dashboard(data)

def main():

    dirName = str(MAP_ID) + "_" + str(wDelay) + "_" + str(wResource) + "_" + str(capRatio) + "_" + str(
        HORIZON) + "_" + str(TOTAL_VESSEL) + "_" + args.agent

    if args.mode == "option":
        drawOption(path="log/"+dirName+"/plots/")
    if args.mode == "general":
        dash(path="log/"+dirName+"/plots/")

    print ("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    