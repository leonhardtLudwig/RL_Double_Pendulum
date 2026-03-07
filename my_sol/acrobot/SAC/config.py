import numpy as np

# ══════════════════════════════════════════════════════
# FISICA
# ══════════════════════════════════════════════════════
DT              = 0.001
INTEGRATOR      = "runge_kutta"
ROBOT           = "acrobot"
MAX_VELOCITY    = 20.0          # rad/s
MODEL_PAR_PATH  = "acrobot_parameters.yml"
GOAL            = np.array([np.pi, 0.0, 0.0, 0.0])

L1 = 0.1
L2 = 0.4

# ══════════════════════════════════════════════════════
# GOAL / RoA — usato da RewardConfig, ComputeRoa, Monitor
# ══════════════════════════════════════════════════════
GOAL_POS_THRESHOLD  = 0.2       # rad
TRAIN_POS_THRESHOLD = 0.3
GOAL_VEL_THRESHOLD  = 0.5       # rad/s
N_HOLD              = 3         # step consecutivi per terminazione

# ══════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════
Q1              = 10.0
Q2              = 10.0
Q3              = 0.1
Q4              = 0.1
R_ACTION        = 1e-4
R_LQR           = 5000.0
REWARD_SCALE    = 1000.0
H_LINE_FRAC     = 0.8           # H_LINE = H_LINE_FRAC * (L1+L2)
R_VEL_NEAR      = 2.0

# ══════════════════════════════════════════════════════
# LQR / RoA
# ══════════════════════════════════════════════════════
# Q_LQR           = np.diag([11.0, 4.0, 0.25, 0.25])
# R_LQR_MAT       = np.diag([0.01, 0.01])

Q_LQR     = np.diag([80.0, 500.0, 20.0, 120.0])
R_LQR_MAT = np.diag([0.2, 0.2]) 

# Q_LQR = np.diag([150.0, 30.0, 7.5, 7.5])
# R_LQR_MAT = np.diag([3, 3])  
# Q_LQR = np.diag([1.92, 1.92, 0.3, 0.3])
# R_LQR_MAT = np.diag([0.82, 0.82])

# Q_LQR     = np.diag([10.0, 10.0, 1.0, 1.0])
# R_LQR_MAT = np.diag([1.0, 1.0])

ROA_N_SAMPLES   = 100
ROA_MAXITER     = 25
ROA_T_SIM       = 10.0
ROA_SUCCESS_RATIO = 0.8
ROA_RHO_MIN     = 1e-3
ROA_RHO_MAX     = 2.0
ROA_S_PATH      = "roa_S.npy"
ROA_RHO_PATH    = "roa_rho.npy"

# ══════════════════════════════════════════════════════
# TRAINING SAC
# ══════════════════════════════════════════════════════
SEED            = 0
STATE_REPR      = 3
MAX_STEPS       = 500
TOTAL_TIMESTEPS = 5_000_000
N_ENVS          = 100
#LEARNING_RATE   = 3e-4
LEARNING_RATE = 0.0001
LEARNING_STARTS = 5_000 #50 x n_envs
GAMMA           = 0.999
TAU             = 0.005
BATCH_SIZE      = 256
ENT_COEF        = 0.1
TARGET_ENTROPY  = -0.5
NET_ARCH        = [256, 256]
N_CRITICS       = 2
#EVAL_FREQ       = 50_000 # TOTAL_TIMESTEPS /20
EVAL_FREQ = int(TOTAL_TIMESTEPS/20)
N_EVAL_EPISODES = 20
PROB_FAR        = 0.5
PROB_NEAR       = 0.3
RESET_VEL       = 1.0
OBS_DIM         = 6 if STATE_REPR == 3 else 4
#BUFFER_SIZE     = max(TOTAL_TIMESTEPS, 100_000)   # almeno 100k, scala col training
#BUFFER_SIZE = 200_000 #0.2 x TOTAL_TIMESTEPS
BUFFER_SIZE = int(0.2 * TOTAL_TIMESTEPS) #ricorda di sistemare
#BUFFER_SIZE = 5_000_000
EP_SECONDS      = MAX_STEPS * DT                  # durata episodio in secondi
RUN_NAME        = "HYBRID"
LOG_DIR_BASE    = "./log_data/SAC_acrobot"

# run notturna
# TOTAL_TIMESTEPS = 150_000_000
# LEARNING_STARTS = 10_000
# EVAL_FREQ = 1_000_000
# BUFFER_SIZE = 2_000_000