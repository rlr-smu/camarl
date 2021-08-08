# ========== Training
AGENT = "diff_mid_hyp"
# AGENT = "appx_dr_colby"  # LOCAL_DR
# AGENT = "appx_dr_scot"   # AT-DR
# AGENT = "mtmf_sm_tgt_lg" # MTMF
# AGENT = "global_count" # MCAC
# AGENT = "ppo_bl" # AT-BASELINE

VERBOSE = False
CSA = "mid"
LOCAL = True
LOAD_MODEL = False
EPISODES = 5
MAP_ID = "map9"

SEED = 1
HORIZON = 200
MAX_AC = 50
VMAX_VMIN = [100, 500]
BATCH_SIZE = 1
ACTIONS = [-0.5, -0.2, 0, 0.2, 0.5]

ARR_RATE = -1
# ARR_RATE = 0.2

# ARR_INTERVAL = [-1]
# ARR_INTERVAL = [2, 3] #0.1
ARR_INTERVAL = [4, 5, 6]  #0.2
# ARR_INTERVAL = [6, 7, 9]  #0.3
# ARR_INTERVAL = [8, 10, 12]  #0.4
# ARR_INTERVAL = [10, 12, 15]  #0.5
# ARR_INTERVAL = [12, 15, 18]  #0.6
# ARR_INTERVAL = [14, 17, 21]  #0.7
# ARR_INTERVAL = [16, 20, 24]  #0.8
# ARR_INTERVAL = [18, 22, 27]  #0.9
# ARR_INTERVAL = [20, 25, 30]  #1.0


CASE = "B"
ALPHA = 0.1
BETA = 0.05
FF = True
ARR_RATE_INTERVAL = 10
MIN_DISTANCE = 3
WIND = True
MAX_WIND = 3

DISCOUNT = 0.99
SHOULD_LOG = 1
SAVE_MODEL = 1
INSTANCE = 1
POLICY_TRAIN = 50
ENV_AC = 0
UPDATE_INTV = 50.0
ENTROPY_WEIGHT = 0

# ========== NN
LEARNING_RATE = 1e-2
GRAD_CLIP = 5
PPO_EPOCH = 2
PPO_CLIP = 0.2
EPOCH = 1
DROPOUT = 0.5
SHUFFLE = True

# ========== Environment
ALT = 2000
NUM_LOS = 6
LOS_NBR = 2

# ========== Diff_Reward
CURR_ARR = MAX_AC

# ========== Others
LARGE = 1000
HUGE = 1e20

# ========== System Environment Variables ========== #
ENV_VAR = True
OPENBLAS = 1
OMP = 1
MKL = 1
NUMEXPR = 1
NUM_CORES = 1
TEST_ID = -1

# ========== Real Data
ADD_TRAFF = 0
MIN_ALT = 5000
MAX_ALT = 10000
RANGE_STR = "5k_10k"
SLICE_INDEX = 10
NEW_NUM_AC = 20

REAL_START_TIME = 6 * 60
REAL_END_TIME = 11 * 60

REAL_DATA_PATH = "/home/james/Codes/data/test/extract"
DATE = "20190501"
DATA_POINT_FREQ = 10
GRID_SIZE = 30