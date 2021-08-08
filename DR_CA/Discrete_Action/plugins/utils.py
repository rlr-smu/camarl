"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Mar 2020
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
from ipdb import set_trace
import numpy as np
from parameters import LARGE, ALPHA, BETA, LOS_NBR, DISCOUNT
import time
import shutil
import matplotlib.pyplot as plt
import io
#plt.style.use('ggplot')
# plt.switch_backend('agg')
#plt.grid()
# plt.ion()

# =============================== Variables ================================== #

# ============================================================================ #
from numba import jit, prange


@jit(nopython=True, fastmath=True)
def discounted_return(reward_list):
    return_so_far = 0
    tmpReturn = []
    for t in range(len(reward_list) - 1, -1, -1):
        return_so_far = reward_list[t] + DISCOUNT * return_so_far
        tmpReturn.append(return_so_far)
    tmpReturn = tmpReturn[::-1]
    return tmpReturn

def discretize_los_nbr(dist_mat):

    # NUM_LOS: 16, NUM_LOS
    # t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
    # t1 = np.where((0 < t1_a) & (t1_a < 3), 0, t1_a)
    # t2 = np.where((3 <= t1) & (t1 < 3.5), 1, t1)
    # t3 = np.where((3.5 <= t2) & (t2 < 4), 2, t2)
    # t4 = np.where((4 <= t3) & (t3 < 4.5), 3, t3)
    # t5 = np.where((4.5 <= t4) & (t4 < 5), 4, t4)
    # t6 = np.where((5 <= t5) & (t5 < 5.5), 5, t5)
    # t7 = np.where((5.5 <= t6) & (t6 < 6), 6, t6)
    # t8 = np.where((6 <= t7) & (t7 < 6.5), 7, t7)
    # t9 = np.where((6.5 <= t8) & (t8 < 7), 8, t8)
    # t10 = np.where((7 <= t9) & (t9 < 7.5), 9, t9)
    # t11 = np.where((7.5 <= t10) & (t10 < 8), 10, t10)
    # t12 = np.where((8 <= t11) & (t11 < 8.5), 11, t11)
    # t13 = np.where((8.5 <= t12) & (t12 < 9), 12, t12)
    # t14 = np.where((9 <= t13) & (t13 < 9.5), 13, t13)
    # t15 = np.where((9.5 <= t14) & (t14 < 10), 14, t14)
    # t16 = np.where(10 <= t15, 15, t15)
    # t17 = np.sort(t16, 1)[:, :LOS_NBR]
    # return t17.astype(np.int)


    # NUM_LOS: 6, NUM_LOS
    t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
    t1 = np.where((0 < t1_a) & (t1_a < 3), 0, t1_a)
    t2 = np.where((3 <= t1) & (t1 < 4), 1, t1)
    t3 = np.where((4 <= t2) & (t2 < 6), 2, t2)
    t4 = np.where((6 <= t3) & (t3 < 8), 3, t3)
    t5 = np.where((8 <= t4) & (t4 < 10), 4, t4)
    t6 = np.where(10 <= t5, 5, t5)
    t7 = np.sort(t6, 1)[:, :LOS_NBR]
    return t7.astype(np.int)


    # NUM_LOS: 5, NUM_LOS
    # t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
    # t1 = np.where((0 < t1_a) & (t1_a < 3), 0, t1_a)
    # t2 = np.where((3 <= t1) & (t1 < 6), 1, t1)
    # t3 = np.where((6 <= t2) & (t2 < 8), 2, t2)
    # t4 = np.where((8 <= t3) & (t3 < 10), 3, t3)
    # t5 = np.where(10 <= t4, 4, t4)
    # t6 = np.sort(t5, 1)[:, :LOS_NBR]
    # return t6.astype(np.int)


    # NUM_LOS: 9, NUM_LOS
    # t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
    # t1 = np.where((0 < t1_a) & (t1_a < 3), 0, t1_a)
    # t2 = np.where((3 <= t1) & (t1 < 4), 1, t1)
    # t3 = np.where((4 <= t2) & (t2 < 5), 2, t2)
    # t4 = np.where((5 <= t3) & (t3 < 6), 3, t3)
    # t5 = np.where((6 <= t4) & (t4 < 7), 4, t4)
    # t6 = np.where((7 <= t5) & (t5 < 8), 5, t5)
    # t7 = np.where((8 <= t6) & (t6 < 9), 6, t6)
    # t8 = np.where((9 <= t7) & (t7 < 10), 7, t7)
    # t9 = np.where(10 <= t8, 8, t8)
    # t10 = np.sort(t9, 1)[:, :LOS_NBR]
    # return t10.astype(np.int)

    # NUM_LOS: 10, NUM_LOS
    # t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
    # t1 = np.where((0 < t1_a) & (t1_a <= 2), 0, t1_a)
    # t2 = np.where((2 < t1) & (t1 <= 4), 1, t1)
    # t3 = np.where((4 < t2) & (t2 <= 6), 2, t2)
    # t4 = np.where((6 < t3) & (t3 <= 8), 3, t3)
    # t5 = np.where((8 < t4) & (t4 <= 10), 4, t4)
    # t6 = np.where((10 < t5) & (t5 <= 12), 5, t5)
    # t7 = np.where((12 < t6) & (t6 <= 14), 6, t6)
    # t8 = np.where((14 < t7) & (t7 <= 16), 7, t7)
    # t9 = np.where((16 < t8) & (t8 <= 18), 8, t8)
    # t10 = np.where(18 < t9, 9, t9)
    # t11 = np.sort(t10, 1)[:, :LOS_NBR]
    # return t11.astype(np.int)

def discretize_los(dist_mat):

        # NUM_LOS : 10
        t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
        t1 = np.where((0 < t1_a) & (t1_a <= 2), 0, t1_a)
        t2 = np.where((2 < t1) & (t1 <= 4), 1, t1)
        t3 = np.where((4 < t2) & (t2 <= 6), 2, t2)
        t4 = np.where((6 < t3) & (t3 <= 8), 3, t3)
        t5 = np.where((8 < t4) & (t4 <= 10), 4, t4)
        t6 = np.where((10 < t5) & (t5 <= 12), 5, t5)
        t7 = np.where((12 < t6) & (t6 <= 14), 6, t6)
        t8 = np.where((14 < t7) & (t7 <= 16), 7, t7)
        t9 = np.where((16 < t8) & (t8 <= 18), 8, t8)
        t10 = np.where(18 < t9, 9, t9)
        t11 = t10.min(1)
        return t11

        # NUM_LOS : 6
        # t1_a = np.where(dist_mat == 0, LARGE, dist_mat)
        # t1 = np.where((0 < t1_a) & (t1_a <= 2), 0, t1_a)
        # t2 = np.where((2 < t1) & (t1 <= 6), 1, t1)
        # t3 = np.where((6 < t2) & (t2 <= 9), 2, t2)
        # t4 = np.where((9 < t3) & (t3 <= 12), 3, t3)
        # t5 = np.where((12 < t4) & (t4 <= 15), 4, t4)
        # t6 = np.where(15 < t5, 5, t5)
        # t7 = t6.min(1)
        # return t7

        # t1 = np.where((0 < dist_mat) & (dist_mat <= 3), 3, dist_mat)
        # t2 = np.where((3 < t1) & (t1 <= 6), 6, t1)
        # t3 = np.where((6 < t2) & (t2 <= 9), 9, t2)
        # t4 = np.where((9 < t3) & (t3 <= 12), 12, t3)
        # t5 = np.where((12 < t4) & (t4 <= 15), 15, t4)
        # t6 = np.where(15 < t5, 19, t5)
        # t7 = np.where(t6 == 0, 19, t6)
        # t8 = t7.min(1)
        # return t8

def display(lg, state):

        # print(" state :")
        # print("", state)
        lg.writeln(" state:")
        lg.writeln(" " +str(state))

class log:

    def __init__(self, fl):
        self.opfile = fl
        if os.path.exists(self.opfile):
            os.remove(self.opfile)
        # f = open(self.opfile, 'w')
        # f.write("test")
        # f.close()

    def writeln(self, msg):
        file = self.opfile
        print(str(msg))
        with open(file, "a") as f:
            f.write("\n"+str(msg))

    def write(self, msg):
        file = self.opfile
        print(str(msg),)
        with open(file, "a") as f:
            f.write(str(msg))

def deleteDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)

# class auxLib:
#
#     def __init__(self):
#         pass
#
#     def file2list(self, fname):
#         assert os.path.exists(fname), "File not found in path : " + fname
#         tmpList = []
#         with open(fname) as f:
#             for line in f:
#                 tmpList.append(line.strip())
#         return tmpList
#
#     def file2y(self, path=None, string=None):
#         tlist = self.file2list(path)
#         y = []
#         for i in range(len(tlist)):
#             line = tlist[i]
#             if string in line:
#                 t1 = line.split(" : ")
#                 y.append(float(t1[1]))
#         return y
#
#     def now(self):
#
#         return time.time()
#
#     def getRuntime(self, st, en):
#
#         return round(en - st, 3)
#
# class plotMovingAverage:
#
#     '''
#     t1 = list(np.random.randint(1, 100, size=100))
#     t2 = list(np.random.randint(1, 100, size=100))
#     data = {}
#     data['a1'] = {}
#     data['a1']['y'] = t1
#     data['a1']['c'] = 'red'
#     data['a2'] = {}
#     data['a2']['y'] = t2
#     data['a2']['c'] = 'blue'
#     pl = plotMovingAverage(data=data, batch_size=5)
#     # pl.show()
#     pl.save("test2.png")
#     '''
#
#     def __init__(self,data=None, batch_size=None, xlabel="", ylabel="", title="", ylim_min=-1, ylim_max=-1, xlim_min=-1, xlim_max=-1):
#         '''
#         :param data: a dict,
#             data['algo1']['y'] = [29, 23, 32]
#             data['algo1']['color'] = 'red'
#         :param batch_size: windown size of moving average, batch_size should properly divide each list.
#         '''
#         self.batch_size = batch_size
#         self.data = data
#         self.xlabel = xlabel
#         self.ylabel = ylabel
#         self.title = title
#         self.xlim_min = xlim_min
#         self.xlim_max = xlim_max
#         self.ylim_min = ylim_min
#         self.ylim_max = ylim_max
#
#     def show(self):
#
#         plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
#         for li in self.data:
#             dt = self.prepData(self.data[li]['y'])
#             x = np.array(np.arange(dt.shape[1]))
#             self.tsplot(plt, x, dt, color=self.data[li]['c'], label=li)
#
#         if self.ylim_min != -1 and self.ylim_max != -1:
#             plt.ylim(bottom=self.ylim_min, top=self.ylim_max)
#         elif self.ylim_min != -1:
#             plt.ylim(bottom=self.ylim_min)
#         elif self.ylim_max != -1:
#             plt.ylim(top=self.ylim_max)
#
#         if self.xlim_min != -1 and self.xlim_max != -1:
#             plt.xlim(bottom=self.xlim_min, top=self.xlim_max)
#         elif self.xlim_min != -1:
#             plt.xlim(bottom=self.xlim_min)
#         elif self.xlim_max != -1:
#             plt.xlim(top=self.xlim_max)
#
#         plt.xlabel(self.xlabel)
#         plt.ylabel(self.ylabel)
#         plt.title(self.title)
#         plt.show()
#
#     def save(self, fname=None):
#
#         plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
#         for li in self.data:
#             dt = self.prepData(self.data[li]['y'])
#             x = np.array(np.arange(dt.shape[1]))
#             self.tsplot(plt, x, dt, color=self.data[li]['c'], label=li)
#
#         if self.ylim_min != -1 and self.ylim_max != -1:
#             plt.ylim(bottom=self.ylim_min, top=self.ylim_max)
#         elif self.ylim_min != -1:
#             plt.ylim(bottom=self.ylim_min)
#         elif self.ylim_max != -1:
#             plt.ylim(top=self.ylim_max)
#
#         if self.xlim_min != -1 and self.xlim_max != -1:
#             plt.xlim(bottom=self.xlim_min, top=self.xlim_max)
#         elif self.xlim_min != -1:
#             plt.xlim(bottom=self.xlim_min)
#         elif self.xlim_max != -1:
#             plt.xlim(top=self.xlim_max)
#
#         plt.xlabel(self.xlabel)
#         plt.ylabel(self.ylabel)
#         plt.title(self.title)
#         plt.savefig(fname)
#
#     def prepData(self, dt):
#
#         dt = np.array(dt)
#         dt = np.array(np.split(dt, dt.shape[0] / self.batch_size))
#         t1 = np.transpose(dt)
#         return t1
#
#     def tsplot(self, plt, x, data,title="", label="", color="b", alpha=0.25,  linestyle="-", **kw):
#         est = np.mean(data, axis=0)
#         sd = np.std(data, axis=0)
#         cis = (est - sd, est + sd)
#         plt.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
#         plt.plot(x,est,label=label, color=color, linestyle=linestyle, **kw)
#         plt.legend()

def real_time(plot_conf, plot_conf_mean, plot_goal_reached):

    # plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)

    # sbplt = int(str(11)+str(1))
    # plt.subplot(sbplt)

    # plt.grid()
    plt.ylabel("Total Conflicts")
    plt.xlabel("Time")
    plt.plot(plot_conf, color='red', label="Total conflicts")
    plt.plot(plot_conf_mean, color='blue', label="Running Avg. conflicts")

    # sbplt = int(str(21)+str(2))
    # plt.subplot(sbplt)
    # plt.ylabel("Goal Reached")
    # plt.xlabel("Time")
    # plt.plot(plot_goal_reached, color='blue')

    plt.draw()
    plt.pause(0.001)

class pred_plot_e():

    def __init__(self, tmp_data=None, ppath=None):

        figcount = 0
        # map2
        # for id in range(2*2):
        #     sbplt = int(str(2)+str(2)+str(id+1))
        # map0
        for id in range(4 * 2):
            sbplt = int(str(2) + str(1) + str(id + 1))
            # self.fig = plt.figure(figcount, figsize=(10, 6))
            self.fig = plt.figure(figcount, figsize=(10, 6))
            plt.subplot(sbplt)
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
            plt.title("Edge_"+str(id))
            plt.plot(tmp_data[id]['target'], color="blue", label="Target")
            plt.plot(tmp_data[id]['pred'], color="red", label="Pred")
            plt.legend()
        # plt.savefig(ppath)
        # self.buf = io.BytesIO()
        # plt.savefig(self.buf, format='png')
        # self.buf.seek(0)

class pred_plot():

    def __init__(self, tmp_data=None, ppath=None):

        figcount = 0
        # map2
        # for id in range(2*2):
        #     sbplt = int(str(2)+str(2)+str(id+1))
        # map6
        # for id in range(4 * 2):
        #     sbplt = int(str(2) + str(1) + str(id + 1))

        sbplt = int(str(1) + str(1) + str(1))
        self.fig = plt.figure(figcount, figsize=(10, 6))
        plt.subplot(sbplt)
        plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
        plt.plot(tmp_data['target'], color="blue", label="Target")
        plt.plot(tmp_data['pred'], color="red", label="Pred")
        plt.legend()


def main():


    print("Hello World")



# =============================================================================== #

if __name__ == '__main__':
    main()
    