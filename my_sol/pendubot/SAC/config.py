import numpy as np

# ══════════════════════════════════════════════════════
# FISICA
# ══════════════════════════════════════════════════════
DT              = 0.01
INTEGRATOR      = "runge_kutta"
ROBOT           = "pendubot"
MAX_VELOCITY    = 20.0          # rad/s
MODEL_PAR_PATH  = "pendubot_parameters.yml"
GOAL            = np.array([np.pi, 0.0, 0.0, 0.0])

L1 = 0.4
L2 = 0.1

# ══════════════════════════════════════════════════════
# GOAL / RoA — usato da RewardConfig, ComputeRoa, Monitor
# ══════════════════════════════════════════════════════
GOAL_POS_THRESHOLD  = 0.2       # rad
GOAL_VEL_THRESHOLD  = 0.5       # rad/s
N_HOLD              = 3         # step consecutivi per terminazione

# ══════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════
Q1              = 150.0
Q2              = 5.0
Q3              = 5.0
Q4              = 5.0
R_ACTION        = 1e-3
R_LQR           = 5000.0
REWARD_SCALE    = 1000.0
H_LINE_FRAC     = 0.8           # H_LINE = H_LINE_FRAC * (L1+L2)
R_VEL_NEAR      = 2.0

# ══════════════════════════════════════════════════════
# LQR / RoA
# ══════════════════════════════════════════════════════
Q_LQR           = np.diag([10.0, 1.0, 1.0, 1.0])
R_LQR_MAT       = np.diag([1.0, 1.0])
ROA_N_SAMPLES   = 100
ROA_MAXITER     = 50
ROA_T_SIM       = 10.0
ROA_SUCCESS_RATIO = 0.85
ROA_RHO_MAX     = 50.0
ROA_S_PATH      = "roa_S.npy"
ROA_RHO_PATH    = "roa_rho.npy"

# ══════════════════════════════════════════════════════
# TRAINING SAC
# ══════════════════════════════════════════════════════
SEED            = 0
STATE_REPR      = 3
MAX_STEPS       = 500
TOTAL_TIMESTEPS = 100_000
N_ENVS          = 1
LEARNING_RATE   = 3e-4
LEARNING_STARTS = 1_000
GAMMA           = 0.999
TAU             = 0.005
BATCH_SIZE      = 256
ENT_COEF        = "auto_0.1"
TARGET_ENTROPY  = -1.0
NET_ARCH        = [256, 256]
N_CRITICS       = 2
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 10
PROB_FAR        = 0.5
PROB_NEAR       = 0.1
RESET_VEL       = 1.0
RUN_NAME        = "V3"
LOG_DIR_BASE    = "./log_data/SAC_pendubot"
