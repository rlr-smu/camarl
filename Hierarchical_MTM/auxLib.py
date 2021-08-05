
import time
import os
import shutil
import pickle

global opFile


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

def tsplot(plt, x, data, label="", color="b", alpha=0.25,  linestyle="-", **kw):
    # x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
    plt.plot(x,est,label=label, color=color, linestyle=linestyle, **kw)
    plt.legend()


def average(tList):

    if len(tList) == 0:
        return 0
    return float(sum(tList))/len(tList)

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

def deleteDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)

def file2list(fname):
    assert os.path.exists(fname), "File not found in path : "+fname
    tmpList = []
    with open(fname) as f:
        for line in f:
            tmpList.append(line.strip())
    return tmpList
