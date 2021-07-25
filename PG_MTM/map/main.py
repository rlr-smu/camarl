"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 10 Jul 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ secImports ================================ #

import sys
import os
import platform
from pprint import pprint
import time

o = platform.system()
if o == "Linux":
    d = platform.dist()
    if d[0] == "debian":
        sys.path.append("/home/james/Codes/python")
    if d[0] == "centos":
        sys.path.append("/home/arambamjs.2016/projects")
    if d[0] == "redhat":
        sys.path.append("/home/arambamjs/projects")
if o == "Darwin":
    sys.path.append("/Users/james/Codes/python")
import auxlib.auxLib as ax

# ================================ priImports ================================ #
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from ctypes.util import find_library
find_library('geos_c')
import seaborn


# ============================================================================ #
print "# ============================ START ============================ #"
# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location

# -------------------------------------------------------------- #

def load(map):

    coord = []
    fig = {}
    with open(map) as f:
        for line in f:
            tmp = line.split(";")
            for line in tmp:
                if line[0:5] == " draw" and "cycle" in line:
                    line = line[6:len(line)]
                    tmp2 = line.split("--")
                    tmp2 = tmp2[0:4]
                    coord.append(tmp2)
    pCount = 0
    for p in coord:
        tmp2 = []
        for c in p:
            tmp = c.split(",")
            x = float(tmp[0][1:len(tmp[0])])
            y = float(tmp[1][0:len(tmp[1])-1])
            tmp2.append((x, y))
        fig[pCount] = tmp2
        pCount += 1
    return fig

def main():

    map = load("2.txt")
    print map[0]
    exit()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in map:
        p = Polygon(map[i])
        x, y = p.exterior.xy
        ax.plot(x, y, 'black')
    plt.show()



# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
