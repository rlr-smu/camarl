
# AGENT = "diff_max"
AGENT = "diff_mid"
# AGENT = "mean_field"
# AGENT = "appx_dr_colby"

EPISODES = 100000
HORIZON = 25
LOCAL = True
LOAD_MODEL = False
# LOAD_MODEL = True
CSA = "mid"
SEED = 4
NUM_AGENTS = 3
NUM_LM = NUM_AGENTS
RENDER = False
NUM_ACTIONS = 5
BATCH_SIZE = 41
SAVE_MODEL = 10 # 100
SHOULD_LOG = 1

# ====== Tile-Coding
FEATURE_RANGE = [[-1, 1], [-1, 1]]
NUM_TILES = 2
GRID = 5
OFFSETS = [[0, 0], [0.2, 1]]

# ===== NN
LEARNING_RATE = 1e-3
EPOCH = 1
DROPOUT = 0.5
SHUFFLE = True
DISCOUNT = 0.99
POLICY_TRAIN = 100
ENTROPY_WEIGHT = 0

# ========== System Environment Variables
ENV_VAR = True
OPENBLAS = 1
OMP = 1
MKL = 1
NUMEXPR = 1
NUM_CORES = 1

# ========== Others
HUGE = 1e5