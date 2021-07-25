"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 02 Jul 2017
Description : Parameters
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ++++++++++++++++++++++++++++++++ Parameters ++++++++++++++++++++++++++++++++ #

# +++++++ Experiment +++++++ #
# AGENT_NAME = "pg_fict"
# AGENT_NAME = "ddpg"
AGENT_NAME = "pg_vanilla"
# AGENT_NAME = "tmin"



EPISODES = 2 #1000 #100000 #100000
SHOULD_LOG = 1 #100
SAMPLE_AVG = 1 #100
GRANULARITY = 1
BATCH_SIZE = 1 #5
TOTAL_RUNTIME = 1 * 15 * 3600
VF_NORM = True
SEED = 1
SAVE_MODEL = 1000
KEEP_MODELS = 1
LOAD_MODEL = False
MODEL_PATH = "./modelPath"
TINY = 1e-8
LARGE = 1e8
MAX_BINARY_LENGTH = 10
RENDER = False
WARM_UP = 300
# +++++++ Synthetic-Instance ++++++ #
INSTANCE = 1
# +++++++ Epsilon-Greedy +++++ #
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 300
# ++++ OrnsteinUhlenbeckProcess
THETA = 0.15
SIGMA = 0.2
MU = 0.0
# ++++ Pytorch ++++ #
NUM_CORES = 1

# ++++++ MAP +++++++ #
TOTAL_VESSEL = 100
MAP_ID = "11_2"
HORIZON = 200
succProb = 0.09
MODES = 2
ARR_HORIZON = 20
BUFFER_MIN = 1
BUFFER_MAX = 20

wResource = 0
wDelay = 1
totCapRatio = 1
capRatio = 0.2
vSize = 1
resMin = 10
resMax = 2 * 10

# +++++++ BASELINES_INF ++++++++ #
BASELINE = "rl_ddpg"
CP_COL_K = 50
# BASELINE_PATH = "./NLP/Cplex"
# BASELINE_PATH = "/home/james/Link to Dropbox/Buffer/ExpResults/RL_Workshop/wR/CP_Lucas/"
BASELINE_PATH = "/home/james/Link to Dropbox/Buffer/ExpResults/RL_Workshop/wR/CP_Collective/K50/"
# BASELINE_PATH = "./cp_col/"
BASELINE_PATH = "./log/"
WORKER = 3


# ++++++ Inference +++++++ #
OLD_HORIZON = 200
# ++++++ CP_SAA +++++++ #
CP_SAA_SAMPLES = 100
CP_TIME_LIMIT = 50
CP_OPTMALITY_TOL = 1e-3
CP_WORKER = 2
# +++++++ Lucas ++++++ #
LUCAS_CP_START = False
T_MIN_LUCAS = [2, 4]
T_MAX_LUCAS = [12, 15]
# ++++++ T_Min_Max +++++++ #
T_MIN = 1
T_MAX = 3
# ++++++ NN +++++++ #
LEARNING_RATE = 1e-3
OPTIMIZER = 'adam'
DISCOUNT = 0.99
# +++++++++++++ Plots ++++++++ #
x_z = [10 + i * 20 for i in range(300)]
y_z = [10 for i in range(162)]
xAxes = [0, 300, 20]
yAxes = [19, 162, 20]
zSize = [20, 20]

# ++++++++++++++++++++++++++++++++++++++++++++ #
