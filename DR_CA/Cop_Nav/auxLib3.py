"""
Author : James Arambam
Date   : 19th March 2016
"""

# ================================================== #

import numpy as np
import networkx as nx
import platform
import pickle
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib.style.use('ggplot')
import seaborn
# from bokeh.plotting import figure, output_server, cursession, show, gridplot
import os
import time
import shutil
from heapq import heappop, heappush
from itertools import count
from random import shuffle
import random as rnd
from multiprocessing import Pool
import multiprocessing
import matplotlib.style as style
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

style.use("seaborn")
import pdb

from ipdb import set_trace


# ================ Libraries ================= #

# -------- Global Variables -------- #
import math
from pprint import pprint

global ds, sbM_figcount, dX, dY, opFile


sbM_figcount = 0



# ---- draw() ---- #
dX = []
dY = []
# ---------------- #
# -------- Methods -------- #


class log:

    def __init__(self, fl):
        self.opfile = fl
        if os.path.exists(self.opfile):
            os.remove(self.opfile)

    def writeln(self, *args):
        file = self.opfile
        msg = ""
        for arg in args:
            msg += str(arg) +" "

        print(str(msg))
        with open(file, "a") as f:
            f.write("\n"+str(msg))

    def write(self, *args):
        file = self.opfile
        msg = ""
        for arg in args:
            msg += str(arg) +" "

        print(str(msg),)
        with open(file, "a") as f:
            f.write(str(msg))

def file2y(path=None, string=None):
    tlist = file2list(path)
    y = []
    for i in range(len(tlist)):
        line = tlist[i]
        if string in line:
            t1 = line.split(" : ")
            # print(line, t1)
            y.append(float(t1[1]))
    return y

def splitList(n=None, list=None):

    '''
    n : #items per sublist
    '''
    my_list = list
    final = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]
    return final

def joinPNG(files, append="b", output="Out.png", n=3):

    t1 = files[0]
    t1 = t1.split("/")[1:-1]
    path = reduce(lambda v1, v2:v1+"/"+v2, t1)
    path = "/" + path
    tmp1 = splitList(n=n, list=files)
    if append =="b":
        h1 = 1
        tmp2 = []
        for item in tmp1:
            ap = "+append"
            tstr = "convert "+ap+" "
            tstr += reduce(lambda v1, v2 : v1+" "+v2, item)
            tstr += " "+path+"/"+str(h1)+".png"
            os.system(tstr)
            rmv = reduce(lambda v1, v2 : v1+" "+v2, item)
            os.system("rm "+rmv)
            tmp2.append(path+"/"+str(h1)+".png")
            h1 += 1
        ap = "-append"
        tstr = "convert "+ap+" "
        tstr += reduce(lambda v1, v2 : v1+" "+v2, tmp2)
        tstr += " "+path+"/"+output
        os.system(tstr)
        rmv = reduce(lambda v1, v2 : v1+" "+v2, tmp2)
        os.system("rm "+rmv)
        return
    elif append == "v":
        ap = "-append"
    elif append == "h":
        ap = "+append"
    tstr = "convert "+ap+" "
    tstr += reduce(lambda v1, v2 : v1+" "+v2, files)
    tstr += " "+output
    os.system(tstr)
    rmv = reduce(lambda v1, v2 : v1+" "+v2, files)
    os.system("rm "+rmv)

def getMatColor():
    matColor = []
    overlap = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
    for j, n in enumerate(sorted(overlap, reverse=True)):
        matColor.append(str(n))
    return matColor

class plotMovingAverage:

    '''
    t1 = list(np.random.randint(1, 100, size=100))
    t2 = list(np.random.randint(1, 100, size=100))
    data = {}
    data['a1'] = {}
    data['a1']['y'] = t1
    data['a1']['c'] = 'red'
    data['a2'] = {}
    data['a2']['y'] = t2
    data['a2']['c'] = 'blue'
    pl = plotMovingAverage(data=data, batch_size=5)
    # pl.show()
    pl.save("test2.png")
    '''

    def __init__(self,data=None, batch_size=None, xlabel="", ylabel="", title="", ylim_min=-1, ylim_max=-1, xlim_min=-1, xlim_max=-1):
        '''
        :param data: a dict,
            data['algo1']['y'] = [29, 23, 32]
            data['algo1']['color'] = 'red'
        :param batch_size: windown size of moving average, batch_size should properly divide each list.
        '''
        self.batch_size = batch_size
        self.data = data
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlim_min = xlim_min
        self.xlim_max = xlim_max
        self.ylim_min = ylim_min
        self.ylim_max = ylim_max

    def show(self):

        plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
        for li in self.data:
            dt = self.prepData(self.data[li]['y'])
            x = np.array(np.arange(dt.shape[1]))
            self.tsplot(plt, x, dt, color=self.data[li]['c'], label=li)

        if self.ylim_min != -1 and self.ylim_max != -1:
            plt.ylim(bottom=self.ylim_min, top=self.ylim_max)
        elif self.ylim_min != -1:
            plt.ylim(bottom=self.ylim_min)
        elif self.ylim_max != -1:
            plt.ylim(top=self.ylim_max)

        if self.xlim_min != -1 and self.xlim_max != -1:
            plt.xlim(bottom=self.xlim_min, top=self.xlim_max)
        elif self.xlim_min != -1:
            plt.xlim(bottom=self.xlim_min)
        elif self.xlim_max != -1:
            plt.xlim(top=self.xlim_max)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.show()

    def save(self, fname=None):

        plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
        for li in self.data:
            dt = self.prepData(self.data[li]['y'])
            x = np.array(np.arange(dt.shape[1]))
            self.tsplot(plt, x, dt, color=self.data[li]['c'], label=li)

        if self.ylim_min != -1 and self.ylim_max != -1:
            plt.ylim(bottom=self.ylim_min, top=self.ylim_max)
        elif self.ylim_min != -1:
            plt.ylim(bottom=self.ylim_min)
        elif self.ylim_max != -1:
            plt.ylim(top=self.ylim_max)

        if self.xlim_min != -1 and self.xlim_max != -1:
            plt.xlim(bottom=self.xlim_min, top=self.xlim_max)
        elif self.xlim_min != -1:
            plt.xlim(bottom=self.xlim_min)
        elif self.xlim_max != -1:
            plt.xlim(top=self.xlim_max)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.savefig(fname)

    def prepData(self, dt):

        # Equal split
        # dt = np.array(dt)
        # dt = np.array(np.split(dt, dt.shape[0] / self.batch_size))

        # Moving split
        dt = np.array(dt)
        last = dt.shape[0] - self.batch_size
        t1 = np.zeros((last+1, self.batch_size))
        for i in range(last+1):
            t1[i] = dt[i:i+self.batch_size]


        t1 = np.transpose(t1)



        return t1

    def tsplot(self, plt, x, data,title="", label="", color="b", alpha=0.25,  linestyle="-", **kw):
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
        plt.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
        plt.plot(x,est,label=label, color=color, linestyle=linestyle, **kw)
        plt.legend()

class remoteServices:
    '''
    usage:
    sr = remoteServices(server="strider", uname="james")
    sr.dwnFile(rpath="/home/james/misc/log.txt")
    sr.dwnDir(rpath="/home/james/misc/testDir")
    sr.upFile(fname="log.txt", rpath="/home/james/")
    sr.upDir(dname="testDir", rpath="/home/james/")
    '''
    def __init__(self, server="", uname=""):
        self.server = server
        self.uname = uname
        self.home = "/home/" + self.uname + "/"
        self.conn = self.uname+"@"+self.server

    def dwnFile(self, lpath="./", rpath=""):
        cmd = "scp "+self.conn+":"+rpath+" "+lpath
        os.system(cmd)

    def dwnDir(self, lpath="./", rpath=""):
        cmd = "scp -r "+self.conn+":"+rpath+" "+lpath
        os.system(cmd)

    def upFile(self, fname="", rpath=""):
        cmd = "scp ./"+fname+" "+self.conn+":"+rpath
        os.system(cmd)

    def upDir(self, dname="", rpath=""):
        cmd = "scp -r ./"+dname+" "+self.conn+":"+rpath
        os.system(cmd)

class dashboard:

    '''Use
    data = {}
    data['path'] = "./"
    for i in range(1, 5):
        data[i] = {}
        data[i]['fname'] = str(i)+".txt"
        data[i]['xlabel'] = "time"
        data[i]['ylabel'] = "score"
        data[i]['title'] = "score"
    d = dashboard(data)
    '''
    def __init__(self, data={}, save_counter=5):

        self.counter = 0
        self.save_counter = save_counter
        self.data = data
        self.num_plots = len(data) - 1
        self.config()
        self.run()

    def config(self):
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.055, wspace=0.14, hspace=0.44)
        self.ax = []

        if self.num_plots == 1:
            sbplots_row = 1
            sbplots_col = 1

        elif self.num_plots == 2:
            sbplots_row = 2
            sbplots_col = 1

        elif self.num_plots == 3 or self.num_plots == 4:
            sbplots_row = 2
            sbplots_col = 2

        elif self.num_plots == 5 or self.num_plots == 6:
            sbplots_row = 2
            sbplots_col = 3

        elif self.num_plots > 6:
            sbplots_row = int(math.ceil(float(self.num_plots/3.0)))
            sbplots_col = 3

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
            plt.savefig(self.data['path']+"/Result.png")


    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=1000)
        plt.show()

def tsplot(plt, x, data, label="", color="b", alpha=0.25,  linestyle="-", **kw):
    # x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
    plt.plot(x,est,label=label, color=color, linestyle=linestyle, **kw)
    plt.legend()

class adjBar:
    '''
    x = [1, 2, 3]
    y = [[40, 20, 50],[90, 60, 50]]
    bar_label = ["sdfdsf", "dsfds"]
    y_err = [[0,0,0], [0,0,0]]
    pl = adjBar(x=x)
    pl.show(y=y, y_err=y_err, bar_label=bar_label)
    '''
    def __init__(self, title="", barWidth=0.2, legend_loc="upper right", xlabel="", ylabel="", x=[], yscale='linear', xlim=[],ylim=[], fsize='large'):


        self.barWidth = barWidth
        self.legend_loc = legend_loc
        plt.title(title, fontsize=fsize)
        plt.xlabel(xlabel, fontsize=fsize)
        self.yscale = yscale
        plt.xticks(np.arange(len(x)), tuple(x), fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.gca().xaxis.grid()
        plt.ylabel(ylabel, fontsize=fsize)
        if len(ylim) > 0:
            plt.ylim(ylim[0], ylim[1])
        if len(xlim) > 0:
            plt.xlim(xlim[0], xlim[1])
        plt.yscale(yscale)
        plt.rc('legend', fontsize=fsize)


    def logError(self, x, del_x):
        return map(lambda i: (0.434 * del_x[i]) / x[i], [_ for _ in range(len(x))])

    def save(self, y=[], bar_label=[], y_err=[], fname="bar.png"):

        total = len(y)
        x = []
        x.append([_ for _ in range(len(y[0]))])
        for i in range(1, total):
            x.append([ v + i * self.barWidth for v in x[0]])
        for i in range(len(y)):
            if self.yscale == 'log':
                err = self.logError(y[i], y_err[i])
                plt.bar(x[i], y[i], yerr=err, width=self.barWidth, label=bar_label[i], capsize=5)
            else:
                plt.bar(x[i], y[i], yerr=y_err[i], width=self.barWidth, label=bar_label[i], capsize=5)
        plt.legend(loc=self.legend_loc)
        plt.savefig(fname)
        plt.close('all')

    def show(self, y=[], bar_label=[], y_err=[]):

        total = len(y)
        x = []
        x.append([_ for _ in range(len(y[0]))])
        for i in range(1, total):
            x.append([ v + i * self.barWidth for v in x[0]])
        for i in range(len(y)):
            if self.yscale == 'log':
                err = self.logError(y[i], y_err[i])
                plt.bar(x[i], y[i], yerr=err, width=self.barWidth, label=bar_label[i], capsize=5)
            else:
                plt.bar(x[i], y[i], yerr=y_err[i], width=self.barWidth, label=bar_label[i], capsize=5)
        plt.legend(loc=self.legend_loc)
        plt.show()
        plt.close('all')

def file2list(fname):
    assert os.path.exists(fname), "File not found in path : "+fname
    tmpList = []
    with open(fname) as f:
        for line in f:
            tmpList.append(line.strip())
    return tmpList

class multiProcess:

    def __init__(self, func, items):

        self.func = func
        self.items = items
        self.cpu = multiprocessing.cpu_count()

    def run(self):

        self.pool = Pool(processes=self.cpu - 1)
        tmpReturn = self.pool.map(self.func, self.items)
        self.pool.close()
        self.pool.join()
        return tmpReturn

def average(tList):

    if len(tList) == 0:
        return 0
    return float(sum(tList))/len(tList)

def createLog():

    global opFile
    # deleteDir("./logs/")
    dir = "./logs/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = opFile
    f = open(dir, 'w')
    f.close()

def writeln(msg):

    global opFile
    file = opFile
    print(str(msg))
    with open(file, "a") as f:
        f.write("\n"+str(msg))

def write(msg):

    global opFile
    file = opFile
    print(str(msg),)
    with open(file, "a") as f:
        f.write(str(msg))

def getOS():

    o = platform.system()
    if o == "Linux":
        d = platform.dist()
        if d[0] == "debian":
            return "ubuntu"
        if d[0] == "centos":
            return "centos"
    if o == "Darwin":
        return "mac"

def loadDataStr(x):

    file2 = open(x+'.pkl', 'rb')
    ds = pickle.load(file2)
    file2.close()
    return ds

def dumpDataStr(x,obj):

    afile = open(x+'.pkl', 'wb')
    pickle.dump(obj, afile)
    afile.close()

def now():

    return time.time()

def getRuntime(st, en):

    return round(en - st, 3)

def createDir(path, folder):

    directory = path + folder
    if not os.path.exists(directory):
        os.makedirs(directory)

def subPlots(x, y, id, **arg):
    sbplt = int(str(42)+str(id))
    plt.subplot(sbplt)
    plt.ylabel(arg.get("yaxis"))
    plt.xlabel(arg.get("xaxis"))
    plt.subplots_adjust(left=0.05, right=0.3, top=0.96, bottom=0.1, wspace= 0.3, hspace = 0.88)
    plt.title(arg.get("title"))
    plt.plot(x, y, label = arg.get("ylabel"))
    if arg.get("z") != None:
        plt.plot(x, arg.get("z"), label = arg.get("zlabel"))
    plt.legend()

def drawG(G, **args):

    pos = nx.random_layout(G)
    nx.draw(G, pos, node_size=100, font_size=20)
    nodes_labels = dict([(u, u) for u in G.nodes()])
    nx.draw_networkx_labels(G, pos, nodes_labels)
    nx.draw(G, pos)
    plt.savefig(args.get("save"), format = "PNG")

def prim_mst_Source(G, s, weight='weight'):

    T = nx.Graph(prim_mst_edges(G, s, weight=weight, data=True))
    if len(T) != len(G):
        T.add_nodes_from([n for n, d in G.degree().items() if d == 0])
    for n in T:
        T.node[n] = G.node[n].copy()
    T.graph = G.graph.copy()
    return T

def prim_mst_edges(G, s, weight='weight', data=True):

    push = heappush
    pop = heappop
    nodes = G.nodes()
    shuffle(nodes)
    nodes.remove(s)
    nodes.insert(0, s)
    c = count()
    temp = []
    while nodes:
        u = nodes.pop(0)
        frontier = []
        visited = [u]
        for u, v in G.edges(u):
            push(frontier, (G[u][v].get(weight, 1), next(c), u, v))
        while frontier:
            W, _, u, v = pop(frontier)
            if v in visited:
                continue
            visited.append(v)
            nodes.remove(v)
            for v, w in G.edges(v):
                if not w in visited:
                    push(frontier, (G[v][w].get(weight, 1), next(c), v, w))
            if data:
                temp.append((u, v))
            else:
                print("####")
    return temp

class seaBplots:

    """
    Description : This class is used for ploting with specified number of subplots.
    x = [i for i in range(10)]
    y = [i * rnd.randint(1, 5) for i in range(10)]
    sb = ax.seaBplots("2x1")
    sb.addPlot(0, "abc",x, y, "foo", "bar")
    sb.addPlot(1, "abc",x, y, "foo", "bar")
    sb.show()
    sb.plot("test2.png")

    For Draw :
    -------------
    import matplotlib.pyplot as plt
    import seaborn
    plt.ion()
    sbm = ax.seaBplots('2x3')
    sbm.addPlot_draw(0, "test0", "episode", "Loss1")
    sbm.addPlot_draw(1, "test1", "episode", "Loss2")

    for i in range(100):
        sbm.update(0, i, rnd.randint(1, 10))
        sbm.update(1, i, 2 * i)
        sbm.draw(plt)
    """

    global cfg, data, fname, R, C, figcount
    cfg = {}
    data = {}
    figcount = 0

    def __init__(self, grid):

        global R, C, figcount
        t = grid.split("x")
        R = int(t[0])
        C = int(t[1])
        figcount += 1

    def addPlot_show(self, spID, title, x, y, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = x
        data[spID]['y'] = y

    def addPlot_draw(self, spID, title, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = []
        data[spID]['y'] = []

    def show(self):

        for id in range(C*R):
            sbplt = int(str(R)+str(C)+str(id+1))
            plt.figure(figcount)
            plt.subplot(sbplt)
            plt.title(cfg[id]['title'])
            plt.xlabel(cfg[id]['xaxis'])
            plt.ylabel(cfg[id]['yaxis'])
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
            plt.plot(data[id]['x'], data[id]['y'])
            plt.legend()
        plt.show()

    def save(self, f):

        for id in range(C*R):
            sbplt = int(str(R)+str(C)+str(id+1))
            plt.figure(figcount, figsize=(20, 10))
            plt.subplot(sbplt)
            plt.title(cfg[id]['title'])
            plt.xlabel(cfg[id]['xaxis'])
            plt.ylabel(cfg[id]['yaxis'])
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
            plt.plot(data[id]['x'], data[id]['y'])
            plt.legend()
        plt.savefig(f, format="PNG")

    def update(self, pid, x, y):

        data[pid]['x'].append(x)
        data[pid]['y'].append(y)

    def draw(self, plt):

        for id in range(C * R):
            sbplt = int(str(C) + str(R) + str(id + 1))
            plt.figure(figcount)  # , figsize=(20, 10)
            plt.subplot(sbplt)
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
            plt.title(cfg[id]['title'])
            plt.xlabel(cfg[id]['xaxis'])
            plt.ylabel(cfg[id]['yaxis'])
            plt.plot(data[id]['x'], data[id]['y'], 'blue')
        plt.legend()
        plt.pause(0.0001)
        plt.draw()

class seaBplotsHist:

    """
    Description : This class is used for ploting with specified number of subplots.
    x = [i for i in range(10)]
    y = [i * rnd.randint(1, 5) for i in range(10)]
    sb = ax.seaBplotsHist("2x1")
    sb.addPlot(0, "abc",x, y, "foo", "bar")
    sb.addPlot(1, "abc",x, y, "foo", "bar")
    sb.show()
    sb.plot("test2.png")

    For Draw :
    -------------
    import matplotlib.pyplot as plt
    import seaborn
    plt.ion()
    sbm = ax.seaBplots('2x3')
    sbm.addPlot_draw(0, "test0", "episode", "Loss1")
    sbm.addPlot_draw(1, "test1", "episode", "Loss2")

    for i in range(100):
        sbm.update(0, i, rnd.randint(1, 10))
        sbm.update(1, i, 2 * i)
        sbm.draw(plt)
    """

    global cfg, data, fname, R, C, figcount
    cfg = {}
    data = {}
    figcount = 0

    def __init__(self, grid):

        global R, C, figcount
        t = grid.split("x")
        R = int(t[0])
        C = int(t[1])
        figcount += 1

    def addPlot_show(self, spID, title, x, y, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = x
        data[spID]['y'] = y

    def addPlot_draw(self, spID, title, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = []
        data[spID]['y'] = []

    def show(self):

        for id in range(C*R):
            sbplt = int(str(R)+str(C)+str(id+1))
            plt.figure(figcount)
            plt.subplot(sbplt)
            plt.title(cfg[id]['title'])
            plt.xlabel(cfg[id]['xaxis'])
            plt.ylabel(cfg[id]['yaxis'])
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
            plt.plot(data[id]['x'], data[id]['y'])
            plt.legend()
        plt.show()

    def save(self, f):

        for id in range(C*R):
            sbplt = int(str(R)+str(C)+str(id+1))
            plt.figure(figcount, figsize=(20, 10))
            plt.subplot(sbplt)
            plt.title(cfg[id]['title'])
            plt.xlabel(cfg[id]['xaxis'])
            plt.ylabel(cfg[id]['yaxis'])
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
            y = data[id]['y']
            bins = np.arange(np.min(y), np.max(y))
            plt.hist(y, bins=bins)
            plt.legend()
        plt.savefig(f, format="PNG")

    def update(self, pid, x, y):

        data[pid]['x'].append(x)
        data[pid]['y'].append(y)

    def draw(self, plt):

        for id in range(C * R):
            sbplt = int(str(C) + str(R) + str(id + 1))
            plt.figure(figcount)  # , figsize=(20, 10)
            plt.subplot(sbplt)
            plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
            plt.title(cfg[id]['title'])
            plt.xlabel(cfg[id]['xaxis'])
            plt.ylabel(cfg[id]['yaxis'])
            plt.plot(data[id]['x'], data[id]['y'], 'blue')
        plt.legend()
        plt.pause(0.0001)
        plt.draw()

class seaBplotsMulti:


    """
    Description : This class is used for ploting multiple plots with fixed 9 subplots each
    sbm = seaBplotsMulti(len(file))
    id = 0
    for i in file:
        y = []
        x = []
        sbm.addPlot(id, str(i), x, y, "Iteration", "Con. Obj")
        id += 1
    sbm.save("test")
    sbm.show()


    # ---- For Draw ----- #
    import matplotlib.pyplot.plt
    import seaborn
    plt.ion()
    sbm = ax.seaBplotsMulti(2)
    sbm.addPlot(0, "test0", "episode", "Loss1")
    sbm.addPlot(1, "test1", "episode", "Loss2")
    for i in range(100):
        sbm.update(0, i, rnd.randint(1, 10))
        sbm.update(1, i, 2*i)
        sbm.draw(plt)


    """
    global tFigs, cfg, data
    cfg = {}
    data = {}

    def __init__(self, tPlot):

        global tFigs, rFig
        tFigs = int(math.ceil(float(tPlot) / 9))

    def addPlot(self, spID, title, x, y, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = x
        data[spID]['y'] = y

    def addPlot_draw(self, spID, title, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = []
        data[spID]['y'] = []

    def save(self, f):

        global sbM_figcount
        C = 3
        R = 3
        for fid in range(tFigs):
            for id in range(C*R):
                sbplt = int(str(R) + str(C) + str(id + 1))
                id += sbM_figcount * 9
                plt.figure(sbM_figcount, figsize=(20,10))
                plt.subplot(sbplt)
                plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
                if id in data:
                    plt.title(cfg[id]['title'])
                    plt.xlabel(cfg[id]['xaxis'])
                    plt.ylabel(cfg[id]['yaxis'])
                    plt.plot(data[id]['x'], data[id]['y'])
                else:
                    plt.title("")
                    plt.xlabel("")
                    plt.ylabel("")
                    plt.plot([], [])
                plt.legend()
            plt.savefig(f+str(fid)+".png", format="PNG")
            sbM_figcount += 1

    def show(self):

        global sbM_figcount
        C = 3
        R = 3
        for fid in range(tFigs):
            for id in range(C * R):
                sbplt = int(str(R) + str(C) + str(id + 1))
                id += sbM_figcount * 9
                plt.figure(sbM_figcount) # , figsize=(20, 10)
                plt.subplot(sbplt)
                plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
                if id in data:
                    plt.title(cfg[id]['title'])
                    plt.xlabel(cfg[id]['xaxis'])
                    plt.ylabel(cfg[id]['yaxis'])
                    plt.plot(data[id]['x'], data[id]['y'])
                else:
                    plt.title("")
                    plt.xlabel("")
                    plt.ylabel("")
                    plt.plot([], [])
                plt.legend()
            plt.show()
        sbM_figcount += 1

    def update(self, pid, x, y):

        data[pid]['x'].append(x)
        data[pid]['y'].append(y)

    def draw(self, plt):
        for fid in range(tFigs):
            for id in range(3 * 3):
                sbplt = int(str(3) + str(3) + str(id + 1))
                id += sbM_figcount * 9
                plt.figure(sbM_figcount)  # , figsize=(20, 10)
                plt.subplot(sbplt)
                plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
                if id in data:
                    plt.title(cfg[id]['title'])
                    plt.xlabel(cfg[id]['xaxis'])
                    plt.ylabel(cfg[id]['yaxis'])
                    plt.plot(data[id]['x'], data[id]['y'], 'blue')
                else:
                    plt.title("")
                    plt.xlabel("")
                    plt.ylabel("")
                    plt.plot([], [])
            plt.legend()
            plt.pause(0.0001)
            plt.draw()

class seaBplotsMulti_Bar:


    """
    Description : This class is used for ploting multiple bar plots with fixed 9 subplots each
    sbm = seaBplotsMulti_Bar(Total_Plots)
    id = 0
    for i in range(Total_Plots):
        y = []
        x = []
        sbm.addPlot(id, str(i), x, y, "Iteration", "Con. Obj")
        id += 1
    sbm.save("test")
    sbm.show()
    """
    global tFigs, cfg, data
    cfg = {}
    data = {}

    def __init__(self, tPlot):

        global tFigs, rFig
        tFigs = int(math.ceil(float(tPlot) / 9))

    def addPlot(self, spID, title, x, y, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = x
        data[spID]['y'] = y

    def save(self, f):

        global sbM_figcount
        C = 3
        R = 3
        for fid in range(tFigs):
            for id in range(C*R):
                sbplt = int(str(R) + str(C) + str(id + 1))
                id += sbM_figcount * 9
                plt.figure(sbM_figcount, figsize=(20,10))
                plt.subplot(sbplt)
                plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
                if id in data:
                    plt.title(cfg[id]['title'])
                    plt.xlabel(cfg[id]['xaxis'])
                    plt.ylabel(cfg[id]['yaxis'])
                    plt.bar(data[id]['x'], data[id]['y'])
                else:
                    plt.title("")
                    plt.xlabel("")
                    plt.ylabel("")
                    plt.bar([], [])
                plt.legend()
            plt.savefig(f+str(fid)+".png", format="PNG")
            sbM_figcount += 1

class seaBplotsMulti_Hist:


    """
    Description : This class is used for ploting multiple bar plots with fixed 9 subplots each
    sbm = seaBplotsMulti_Hist(Total_Plots)
    id = 0
    for i in range(Total_Plots):
        y = []
        x = []
        sbm.addPlot(id, str(i), x, y, "Iteration", "Con. Obj")
        id += 1
    sbm.save("test")
    sbm.show()
    """
    global tFigs, cfg, data
    cfg = {}
    data = {}

    def __init__(self, tPlot):

        global tFigs, rFig
        tFigs = int(math.ceil(float(tPlot) / 9))

    def addPlot(self, spID, title, x, y, xaxis, yaxis):

        global cfg, data
        cfg[spID] = {}
        cfg[spID]["title"] = title
        cfg[spID]["xaxis"] = xaxis
        cfg[spID]["yaxis"] = yaxis
        data[spID] = {}
        data[spID]['x'] = x
        data[spID]['y'] = y

    def save(self, f):

        global sbM_figcount
        C = 3
        R = 3
        for fid in range(tFigs):
            for id in range(C*R):
                sbplt = int(str(R) + str(C) + str(id + 1))
                id += sbM_figcount * 9
                plt.figure(sbM_figcount, figsize=(20,10))
                plt.subplot(sbplt)
                plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
                if id in data:
                    plt.title(cfg[id]['title'])
                    plt.xlabel(cfg[id]['xaxis'])
                    plt.ylabel(cfg[id]['yaxis'])
                    y = data[id]['y']
                    bins = np.arange(np.min(y), np.max(y))
                    plt.hist(y, bins=bins)
                else:
                    plt.title("")
                    plt.xlabel("")
                    plt.ylabel("")
                    plt.bar([], [])
                plt.legend()
            plt.savefig(f+str(fid)+".png", format="PNG")
            sbM_figcount += 1

    def show(self):

        global sbM_figcount
        C = 3
        R = 3
        for fid in range(tFigs):
            for id in range(C * R):
                sbplt = int(str(R) + str(C) + str(id + 1))
                id += sbM_figcount * 9
                plt.figure(sbM_figcount) # , figsize=(20, 10)
                plt.subplot(sbplt)
                plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
                if id in data:
                    plt.title(cfg[id]['title'])
                    plt.xlabel(cfg[id]['xaxis'])
                    plt.ylabel(cfg[id]['yaxis'])
                    y = data[id]['y']
                    bins = np.arange(np.min(y), np.max(y))
                    plt.hist(y, bins=bins)
                else:
                    plt.title("")
                    plt.xlabel("")
                    plt.ylabel("")
                    plt.plot([], [])
                plt.legend()
            plt.show()
        sbM_figcount += 1

def listFiles(path):

    return os.listdir(path)

def getDictnKey(d, val):

    tmp = d.values()
    tmp2 = d.keys()
    tmp3 = {}
    for i in range(0, len(tmp)):
        tmp3[tmp[i]] = tmp2[i]
    return tmp3[val]

def reversDictn(d):

    tmp = d.values()
    tmp2 = d.keys()
    tmp3 = {}
    for i in range(0, len(tmp)):
        tmp3[tmp[i]] = tmp2[i]
    return tmp3

def getLineNo(file, str1):

    l = []
    with open(file) as f:
        for i, st in enumerate(f):
            if str1 in st:
                l.append(i)
    return l

def scrapData(file, str1):

    y = []
    with open(file) as f:
        for i, st in enumerate(f):
            if str1 in st:
                y.append(st)
    return y

def copy(file, src, dst):

    file = src + "/" + file
    shutil.copy(file, dst)

def cartesian(lists):

    """
    t = [[0, 1], [0, 1]]
    print(cartesian(t))
    """
    if lists == []: return [()]
    return [x + (y,) for x in cartesian(lists[:-1]) for y in lists[-1]]

def plotProb(P):

    """
    y = [0.1, 0.2, 0.3]
    plotProb(y)
    """
    x = [i for i in range(len(P))]
    plt.ylabel("P(x)")
    plt.xlabel("x")
    plt.xticks(x)
    plt.bar(x, P, width=0.09)
    plt.show()

def getFileName(tstr):

    tmp1 = tstr.split("/")
    fname = tmp1[len(tmp1)-1]
    fname = tmp1[len(tmp1)-1]
    return fname

def getFilePath(tstr):

    tmp1 = tstr.split("/")
    tmp2 = [tmp1[i] for i in range(0, len(tmp1)-1)]
    return reduce(lambda v1, v2 : v1+"/"+v2, tmp2)

def show(file, tstr, title, xaxis, yaxis):

    line = scrapData(file, tstr)
    y = map(lambda item: float(item.split(":")[1].strip()), line)
    x = [i for i in range(len(y))]
    sb = seaBplots("1x1")
    sb.addPlot(0, title, x, y, xaxis, yaxis)
    sb.show()

def maxKey(d):

    return max(d, key=d.get)

def minKey(d):

    return min(d, key=d.get)

def coinToss(p):

    r = np.random.uniform(0, 1)
    if r < p:
        return 1
    else:
        return 0

def draw(plt, x, y):

    """
    import matplotlib.pyplot as plt
    import seaborn
    plt.ion()
    draw(plt, x, y)
    """
    dX.append(x)
    dY.append(y)
    sbplt = int(str(1) + str(1) + str(1))
    plt.figure(0)  # , figsize=(20, 10)
    plt.subplot(sbplt)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.09, wspace=0.2, hspace=0.9)
    plt.plot(dX, dY, 'blue')
    #plt.plot(dX, dY)
    plt.pause(0.0001)
    plt.draw()

def show(x, y):

    sb = seaBplots("1x1")
    sb.addPlot(0, "", x, y, "", "")
    sb.show()

def emptyDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)
    os.system("mkdir "+dir)

def copyDir(src, dest):

    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)

def deleteDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)


# ======================================== #


def main():
    Total_Plots = 4
    sbm = seaBplotsMulti_Hist(Total_Plots)
    id = 0
    for i in range(Total_Plots):
        x = [i for i in range(10)]
        y = [i * rnd.randint(1, 5) for i in range(10)]
        sbm.addPlot(id, str(i), x, y, "Iteration", "Con. Obj")
        id += 1
    sbm.save("test")



# ======================================== #

if __name__ == '__main__':main()
