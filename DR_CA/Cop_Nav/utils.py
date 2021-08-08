"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 18 Jul 2020
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
import pdb
import rlcompleter
import numpy as np
from parameters import FEATURE_RANGE, NUM_TILES, OFFSETS, GRID, DISCOUNT
import shutil
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

# =============================== Variables ================================== #


# ============================================================================ #


def deleteDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)

def discounted_return(reward_list):
    return_so_far = 0
    tmpReturn = []
    for t in range(len(reward_list) - 1, -1, -1):
        return_so_far = reward_list[t] + DISCOUNT * return_so_far
        tmpReturn.append(return_so_far)
    tmpReturn = tmpReturn[::-1]
    return tmpReturn

class tile_code:

    def __init__(self):

        self.grid_hash = np.zeros((NUM_TILES, GRID, GRID), dtype=int)
        for tl in range(NUM_TILES):
            c = 0
            for i in range(GRID):
                for j in range(GRID):
                    self.grid_hash[tl][i][j] = c
                    c += 1

    def create_tile_1D(self, feat_range, bins, offset):
        return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset

    def create_tiles_2D(self, feature_ranges, number_tilings, bins, offsets):
        tilings = []
        for tile_i in range(number_tilings):
            tiling_bin = bins[tile_i]
            tiling_offset = offsets[tile_i]
            tiling = []
            for feat_i in range(len(feature_ranges)):
                feat_range = feature_ranges[feat_i]
                feat_tiling = self.create_tile_1D(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
                feat_tiling = list(map(lambda x : round(x, 3), feat_tiling))
                tiling.append(feat_tiling)
            tilings.append(tiling)
        self.tilings = np.array(tilings)
        return self.tilings

    def get_tile_codes(self, feature):
        """
        feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
        tilings: tilings with a few layers
        return: the encoding for the feature on each layer
        """
        tilings = self.tilings
        num_dims = len(feature)
        feat_codings = []
        for tiling in tilings:
            feat_coding = []
            for i in range(num_dims):
                feat_i = feature[i]
                tiling_i = tiling[i]  # tiling on that dimension
                coding_i = np.digitize(feat_i, tiling_i)
                feat_coding.append(coding_i)
            feat_codings.append(feat_coding)
        return np.array(feat_codings)


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

def main():

    bins = [[GRID, GRID], [GRID, GRID]]
    tc = tile_code()
    tc.create_tiles_2D(FEATURE_RANGE, NUM_TILES, bins, OFFSETS)
    pos = [0.1, 2.5]
    pos = tc.get_tile_codes(pos)


    print(pos)
    set_trace()



if __name__ == '__main__':

    main()