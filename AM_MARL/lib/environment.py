"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 18 Feb 2019
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
import numpy as np
import pdb
import rlcompleter
import shapefile as shp
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from utils import centroid, getOffset
from auxLib import loadDataStr, listFiles
from parameters import DATA_OUTPUT_PATH, DATE, MONTH, TMIN, TMAX, EXTRA_VESSELS
import numpy as np
from performanceEval import performance
from genTrajectory import distribution
from datetime import datetime
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #
REAL_ZONES = [_ for _ in range(1, 24)]
REAL_ZONES.append(53)
REAL_ZONES.append(54)
hxGrid = loadDataStr("data/overlays/hexgrid/hxGrid")

# ============================================================================ #

class Data:

    def __init__(self):

        dpath = DATA_OUTPUT_PATH+MONTH+"/"+DATE+"/Vessels"
        vesselList = listFiles(dpath)
        self.vessels_all = {}
        for v in vesselList:
            vname = v[:-4]
            tmp = np.genfromtxt(dpath + "/" + v, delimiter=',')
            offsetIndex = getOffset(tmp[0][0])
            if offsetIndex <> -1:
                if len(tmp.shape) == 1:
                    gridList = [-2 for _ in range(offsetIndex)]
                    gridList.extend([tmp[8]])
                    self.vessels_all[vname] = gridList
                else:
                    gridList = [-2 for _ in range(offsetIndex)]
                    tmp = tmp[:, [8]] # because hexgrid ID is 8
                    tmp = tmp.ravel().tolist()
                    gridList.extend(tmp)
                    self.vessels_all[vname] = gridList

        self.vessels = np.zeros((TMAX-TMIN+1, len(self.vessels_all)))
        self.vessels.fill(-1)
        tid = 0
        for t in range(TMIN, TMAX):
            vid = 0
            for v in self.vessels_all:
                if t < len(self.vessels_all[v]):
                    self.vessels[tid][vid] = self.vessels_all[v][t]
                vid += 1
            tid += 1
        # Extra Vessels Traffic
        d = distribution()
        traj = []
        for v in range(len(EXTRA_VESSELS)):
            traj.append(d.trajectory(EXTRA_VESSELS[v], TMIN, TMAX))
        traj2 = np.zeros((TMAX-TMIN+1, len(EXTRA_VESSELS)))
        traj2.fill(-1)
        for t in range(TMAX-TMIN+1):
            for v in range(len(EXTRA_VESSELS)):
                traj2[t][v] = traj[v][t]
        self.vessels = np.append(self.vessels, traj2, axis=1)



class synEnv:

    def __init__(self, mode=""):

        print " > Populating data..."
        if mode == "planning":
            self.data = planData()
        else:
            self.data = Data()
        self.perf = performance()
        # self.plot()

    def reset(self):

        return self.data.vessels[0]

    def act(self, t, action):

        self.perf.closeQuarterCount(t, action, self.data.vessels[t+1])
        return self.data.vessels[t+1]

    # def topology(self, plt):
    #     plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.04, wspace=0.2, hspace=0.9)
    #     plt.xlim(103.52, 104.07)
    #     plt.ylim(0.963, 1.47)
    #     plt.axis('off')
    #     plt.grid(False)
    #
    #     # ----------- HexGrid -------------- #
    #     # for id in hxGrid:
    #     #     x = hxGrid[id]['x']
    #     #     y = hxGrid[id]['y']
    #     #     plt.plot(x, y, color='black', linewidth=0.2)
    #
    #     # --------- TSS -------- #
    #     sf = shp.Reader("data/overlays/tss/tss.shp")
    #     for shape in sf.shapeRecords():
    #         x = [i[0] for i in shape.shape.points[:]]
    #         y = [i[1] for i in shape.shape.points[:]]
    #         plt.plot(x, y, color='black', linewidth=1.5)
    #
    #     # --------- Land Areas ------- #
    #     sf = shp.Reader("./data/overlays/landpolygon/landPolygons.shp")
    #     for shape in sf.shapeRecords():
    #         x = [i[0] for i in shape.shape.points[:]]
    #         y = [i[1] for i in shape.shape.points[:]]
    #         plt.plot(x, y, color='black', linewidth=1.5)

    def plot(self):
        plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.04, wspace=0.2, hspace=0.9)
        plt.axis('off')
        # plt.xlim(103.506, 104.139)
        # plt.ylim(0.963, 1.47)
        plt.grid(False)

        # ----------- HexGrid -------------- #
        hxGrid = loadDataStr("data/overlays/hexgrid/hxGrid")
        for id in hxGrid:
            x = hxGrid[id]['x']
            y = hxGrid[id]['y']
            plt.plot(x, y, color='black', linewidth=0.2)
            xc = hxGrid[id]['centroid'][0]
            yc = hxGrid[id]['centroid'][1]
            plt.text(xc, yc, str(id))


        # ----------- TSS -------------- #
        sf = shp.Reader("data/overlays/tss/tss.shp")
        fields = [field[0] for field in sf.fields[1:]]
        for shape in sf.shapeRecords():
            atr = dict(zip(fields, shape.record))
            id = int(atr['id'])
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x, y, color='navy', linewidth=1.5)

        # --------- Land Areas ------- #
        # sf = shp.Reader("data/overlays/landpolygon/landPolygons.shp")
        # for shape in sf.shapeRecords():
        #     x = [i[0] for i in shape.shape.points[:]]
        #     y = [i[1] for i in shape.shape.points[:]]
        #     plt.plot(x, y, color='black', linewidth=0.4)
        plt.show()

    def save_CC_encounters(self):

        gp_coord = {}
        gp_coord['x'] = []
        gp_coord['y'] = []

        g_coord = {}
        g_coord['x'] = []
        g_coord['y'] = []

        for p in self.perf.snapshot:
            g = p[0]
            gp = p[1]
            g_coord['x'].append(hxGrid[g]['centroid'][0])
            g_coord['y'].append(hxGrid[g]['centroid'][1])
            gp_coord['x'].append(hxGrid[gp]['centroid'][0])
            gp_coord['y'].append(hxGrid[gp]['centroid'][1])

        for fid in range(len(g_coord['x'])):
            plt.clf()
            # ---------- #
            plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.04, wspace=0.2, hspace=0.9)
            plt.xlim(103.52, 104.07)
            plt.ylim(0.963, 1.47)

            plt.axis('off')
            plt.grid(False)

            # --------- TSS -------- #
            sf = shp.Reader("data/overlays/tss/tss.shp")
            for shape in sf.shapeRecords():
                x = [i[0] for i in shape.shape.points[:]]
                y = [i[1] for i in shape.shape.points[:]]
                plt.plot(x, y, color='black', linewidth=1.5)

            # --------- Land Areas ------- #
            sf = shp.Reader("./data/overlays/landpolygon/landPolygons.shp")
            for shape in sf.shapeRecords():
                x = [i[0] for i in shape.shape.points[:]]
                y = [i[1] for i in shape.shape.points[:]]
                plt.plot(x, y, color='black', linewidth=1.5)

            # ---------- #
            plt.scatter(g_coord['x'][fid], g_coord['y'][fid], color='red', s=15)
            plt.scatter(gp_coord['x'][fid], gp_coord['y'][fid], color='black', s=15)
            plt.savefig("logs/test"+str(fid)+".png")

class draw:

    def __init__(self):
        self.data = Data()
        plt.figure(figsize=(40, 30))
        plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.04, wspace=0.2, hspace=0.9)
        self.plotData = {}
        self.testData_trace = {}
        self.testData_trace['x'] = []
        self.testData_trace['y'] = []
        for t in range(TMIN, TMAX + 1):
            self.plotData[t] = {}
        for t in range(TMIN, TMAX + 1):
            self.plotData[t]['x'] = []
            self.plotData[t]['y'] = []
        for t in range(TMIN, TMAX + 1):
            for v in self.data.vessels_all:
                if t < len(self.data.vessels_all[v]):
                    g = self.data.vessels_all[v][t]
                    if g <> -1 and g <> -2:
                        x = hxGrid[g]['centroid'][0]
                        y = hxGrid[g]['centroid'][1]
                        self.plotData[t]['x'].append(x)
                        self.plotData[t]['y'].append(y)

    def step(self, tp, action):

        testVessel_grid = action['grid'][tp]
        testVessel_x = hxGrid[testVessel_grid]['centroid'][0]
        testVessel_y = hxGrid[testVessel_grid]['centroid'][1]
        self.testData_trace['x'].append(testVessel_x)
        self.testData_trace['y'].append(testVessel_y)


        t = tp + TMIN
        plt.ion()
        plt.clf()
        self.topology(plt)
        if t in self.plotData.keys():
            plt.scatter(self.plotData[t]['x'], self.plotData[t]['y'], color='black', s=20)
            plt.scatter(testVessel_x, testVessel_y, color='red', s=30)
            plt.plot(self.testData_trace['x'], self.testData_trace['y'], color='red', linestyle='dashed', linewidth=1)
            plt.draw()
            plt.pause(0.2)

    def final_snapshot(self):

        plt.clf()
        self.topology(plt)
        plt.plot(self.testData_trace['x'], self.testData_trace['y'], color='red', linestyle='dashed', linewidth=1)
        plt.savefig("logs/final" + ".png")

    def onlyHistorical(self):
        plt.ion()
        for t in range(TMIN, TMAX + 1):
            plt.clf()
            self.topology(plt)
            if t in self.plotData.keys():
                plt.scatter(self.plotData[t]['x'], self.plotData[t]['y'], color='black', s=20)
                plt.draw()
                plt.pause(0.2)

    def topology(self, plt):
        plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.04, wspace=0.2, hspace=0.9)
        plt.xlim(103.52, 104.07)
        plt.ylim(0.963, 1.47)
        plt.axis('off')
        plt.grid(False)

        # ----------- HexGrid -------------- #
        # for id in hxGrid:
        #     x = hxGrid[id]['x']
        #     y = hxGrid[id]['y']
        #     plt.plot(x, y, color='black', linewidth=0.2)

        # --------- TSS -------- #
        sf = shp.Reader("data/overlays/tss/tss.shp")
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x, y, color='black', linewidth=1.5)

        # --------- Land Areas ------- #
        sf = shp.Reader("./data/overlays/landpolygon/landPolygons.shp")
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x, y, color='black', linewidth=1.5)

def main():

    # synEnv()


    print "Hello World"


# =============================================================================== #

if __name__ == '__main__':
    main()



