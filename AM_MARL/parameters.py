EPISODES = 1000 #100000
HORIZON = 20
SEED = 1
VERBOSE = False
LOAD_MODEL = False
# LOAD_MODEL = True
SHOULD_LOG = 1
SAVE_MODEL = 100
POLICY_TRAIN = 100
BATCH_SIZE = 10 #512
PY_ENV = "/home/james/miniconda3/envs/am_marl/bin/python"

# ===== Agents
# AGENT_NAME = "random"
# AGENT_NAME = 'vpg'
# AGENT_NAME = 'vpg_dec'
# AGENT_NAME = 'vpg_mem'

# AGENT_NAME = 'idv'
# AGENT_NAME = 'idv_mem'

AGENT_NAME = 'vpg_dec'
# AGENT_NAME = 'idv_dec'
# AGENT_NAME = 'idv_dec_mem'

# ============== Environment
ENV_NAME = 'grid_nav'
TOTAL_AGENTS = 50
GRID = 5
CAP = 5


# ===== Algorithm
NB_DIM = 30
DIST_THRESHOLD = 0.9999
MAX_ITR_VP = 100
LAMBDA = 0.2
NEAREST_NBR = 10


# ===== Multigrid
STATE_GRID_SIZE = 3
# RENDER = True
RENDER = False
# Multiple of state grid size
MID_WALL = True
# MID_WALL = False
SAVE_GRAPH_FIG = False
# SAVE_GRAPH_FIG = True

# ===== collectiveGridNavigation


# ===== NN
DISCOUNT = 0.99
HIDDEN_DIM = 32
LEARNING_RATE = 1e-4
ENTROPY_WEIGHT = 0

# ===== Others
LARGE = 1e10
TINY = 1e-8

# ========== System Environment Variables
ENV_VAR = True
OPENBLAS = 1
OMP = 1
MKL = 1
NUMEXPR = 1
NUM_CORES = 1
