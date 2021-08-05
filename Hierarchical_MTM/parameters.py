SEED = 1

# +++++++ New parameters ++++++ #
# AGENT = 'pg_fict_dcp'
# AGENT = 'op_ind'
AGENT = 'op_pgv'


# +++++++ Experiment +++++++ #
EPISODES = 2000
SHOULD_LOG = 1
SAMPLE_AVG = 1
BATCH_SIZE = 1
TOTAL_RUNTIME = 1 * 24 * 3600
LEARNING_RATE = 1e-3
OPTIMIZER = 'adam'
SAVE_MODEL = 1
LOAD_MODEL = False


# +++++++ Synthetic-Instance ++++++ #
INSTANCE = 1
wDelay = 1
wResource = 0

# +++++++++ Discrete Settings +++++++++ #
NUM_ACTIONS = 10

# +++++++++ MAP +++++++++ #
TOTAL_VESSEL = 1
MAP_ID = "3_1"
HORIZON = 30

# Travel Time
T_MIN = 1
T_MAX_MIN = 2
T_MAX_MAX = 3
RANDOM_TRAVEL = True
BUFFER_MIN = 1
BUFFER_MAX = 2

# Capacity
capRatio = 0.1
BUFFER_CAP = 1e4
BUFFER_VIO = False
RANDOM_CAP = True
RANDOM_CAP_MIN = 5
RANDOM_CAP_MAX = 100

# Arrival Time
MODES = 1
ARR_HORIZON = 2
NUM_ARR_DIST = 1

# =========== Others ========== #
EVAL_EPISODES = 3
EVAL_DATE = "2017-01-01"
DPATH = "synData/real_map"
NUM_OPTIONS = 2
DENSITY = 0.8
ENTROPY_WEIGHT = 0
succProb = 0.09
resMin = 10
resMax = 2 * 10
vSize = 1

# Real Map Stats
REAL_HORIZON = HORIZON
T_MIN_PERCENTILE = 10
T_MAX_PERCENTILE = 90
REAL_START_TIME = 180
REAL_END_TIME = 360
INTERVAL = 60 # in Seconds (Simulation Time unit or Model time unit)
INTERVAL_HORIZON = 60 # The real interval (minutes)
# TOTAL_HORIZON = 1 * 60
EVAL_DURATION = 3 * 60
REAL_HORIZON_EVAL = EVAL_DURATION
EVAL_HORIZON = EVAL_DURATION
EVAL_PEAK_HORIZON = 60
HOUR_START = 4

# +++++++ Not Important ++++++ #
RUN_MODE = "Test"
GRANULARITY = 1
VF_NORM = True
KEEP_MODELS = 1
TINY = 1e-8
LARGE = 1e8
MAX_BINARY_LENGTH = 10
RENDER = False

# ++++ Pytorch ++++ #
WEIGHT_VARIANCE = 0.02
GRADIENT_CLIP = 5
NUM_CORES = 1
DROPOUT = 0.6
DISCOUNT = 0.99
CLIP = 5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# +++++++ System Environment Variables ++++++ #
ENV_VAR = True
OPENBLAS = 1
OMP = 1
MKL = 1
NUMEXPR = 1

