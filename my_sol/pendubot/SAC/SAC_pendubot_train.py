import os
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import CustomEnv, double_pendulum_dynamics_func

#from RewardConfiguration import reward_func, terminated_func, make_noisy_reset_func
#from RewardConfiguration_Paper import make_reward_func, make_terminated_func, make_noisy_reset_func
from RewardConfiguration_V3 import make_reward_func, make_terminated_func, make_noisy_reset_func

from TrainingMonitor import TrainingMonitorCallback

from ComputeRoa import compute_and_save_roa

# SAC_pendubot_train.py
from config import (
    SEED, DT, INTEGRATOR, ROBOT, STATE_REPR, MAX_VELOCITY,
    MAX_STEPS, TOTAL_TIMESTEPS, LEARNING_RATE, GAMMA, TAU,
    BATCH_SIZE, ENT_COEF, TARGET_ENTROPY, NET_ARCH, N_CRITICS,
    EVAL_FREQ, N_EVAL_EPISODES, RUN_NAME, LOG_DIR_BASE,
    ROA_S_PATH, ROA_RHO_PATH, N_ENVS, LEARNING_STARTS
)



# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI — tocca solo questa sezione
# ══════════════════════════════════════════════════════════════════════════════

# SEED = 0

# # simulazione
# DT                  = 0.01
# INTEGRATOR          = "runge_kutta"
# ROBOT               = "pendubot"
# STATE_REPR          = 3          # 2 = angoli diretti (4D), 3 = cos/sin (6D)
# MAX_VELOCITY        = 20.0       # [rad/s] — usato per normalizzazione

# # episodio
# MAX_STEPS           = 500       # steps per episodio (secondi = MAX_STEPS * DT)

# # training
# TOTAL_TIMESTEPS     = 100_000
# N_ENVS              = 1
# LEARNING_STARTS     = 1_000       # inizia ad aggiornare prima

# # SAC
# LEARNING_RATE       = 3e-4
# GAMMA               = 0.99
# TAU                 = 0.005
# BATCH_SIZE          = 256
# ENT_COEF            = "auto_0.1"
# ENT_TARGET          = -0.5
# NET_ARCH            = [256, 256]
# N_CRITICS           = 2

# # evaluation callback
# EVAL_FREQ           = 10_000     # ogni quanti steps valutare
# N_EVAL_EPISODES     = 10
# #notturno 
# # EVAL_FREQ       = 50_000    # ogni 50k steps — non ogni 10k su run lungo
# # N_EVAL_EPISODES = 20        # più episodi = stima più stabile


# # logging
# RUN_NAME            = "v1_cos_reward_random_reset"
#LOG_DIR             = os.path.join("./log_data/SAC_pendubot", RUN_NAME)

model_par_path = "pendubot_parameters.yml"


# roa

ROA_S_PATH   = os.path.join("./", "roa_S.npy")
ROA_RHO_PATH = os.path.join("./", "roa_rho.npy")

if os.path.exists(ROA_S_PATH) and os.path.exists(ROA_RHO_PATH):
    print("RoA già calcolata, carico da file...")
    S_lqr = np.load(ROA_S_PATH)
    rho   = float(np.load(ROA_RHO_PATH)[0])
else:
    print("Calcolo RoA...")
    S_lqr, rho = compute_and_save_roa(
        model_par_path=model_par_path,
        save_S_path=ROA_S_PATH,
        save_rho_path=ROA_RHO_PATH,
        verbose=True,
    )

#ELIMINARE, SOLO PER CONTROLLO
#rho = 1.5

S_lqr = None
rho = None

# print(f"RoA: rho = {rho:.4f}")
# print("S_lqr:")
# print(S_lqr)



# ── parametri derivati (calcolati automaticamente) ─────────────────────────────
OBS_DIM         = 6 if STATE_REPR == 3 else 4
BUFFER_SIZE     = max(TOTAL_TIMESTEPS, 100_000)   # almeno 100k, scala col training
#BUFFER_SIZE = 500_000
EP_SECONDS      = MAX_STEPS * DT                  # durata episodio in secondi

# ══════════════════════════════════════════════════════════════════════════════
# SEED — fisso ovunque
# ══════════════════════════════════════════════════════════════════════════════

set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# MODELLO
# ══════════════════════════════════════════════════════════════════════════════

mpar = model_parameters(filepath=model_par_path)
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)


# ── wrapper float32 ────────────────────────────────────────────────────────────
class DynamicsFuncWrapper:
    def __init__(self, dynamics_func):
        self.dynamics_func = dynamics_func
        self.max_velocity = dynamics_func.max_velocity

    def __call__(self, state, action, scaling=True):
        return self.dynamics_func(state, action, scaling=scaling).astype(np.float32)

    def normalize_state(self, state):
        return self.dynamics_func.normalize_state(state).astype(np.float32)

    def unscale_state(self, state):
        return self.dynamics_func.unscale_state(state)

    def unscale_action(self, action):
        return self.dynamics_func.unscale_action(action)


dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=simulator,
        dt=DT,
        integrator=INTEGRATOR,
        robot=ROBOT,
        state_representation=STATE_REPR,
        max_velocity=MAX_VELOCITY,
        torque_limit=mpar.tl,
        scaling=True,
    )
)

obs_space = spaces.Box(np.array([-1.0]*OBS_DIM), np.array([1.0]*OBS_DIM), dtype=np.float32)
act_space = spaces.Box(np.array([-1.0]),          np.array([1.0]),          dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# REWARD / TERMINATED / RESET
# ══════════════════════════════════════════════════════════════════════════════

# def reward_func(observation, action):
#     cos_th1 = observation[0]
#     cos_th2 = observation[2]
#     dth1 = observation[4] * MAX_VELOCITY
#     dth2 = observation[5] * MAX_VELOCITY

#     r = (
#         + (cos_th1 - 1.0)
#         - 0.3  * (cos_th2 - 1.0)**2
#         - 0.2  * (dth1**2 + dth2**2) / MAX_VELOCITY**2
#         - 0.05 * float(action[0])**2
#     )

#     # bonus goal: vicino a π con basse velocità
#     if cos_th1 < -0.95 and abs(dth1) < 1.0 and abs(dth2) < 1.0:
#         r += 20.0

#     return float(r)


# def terminated_func(observation):
#     cos_th1 = observation[0]
#     dth1 = observation[4] * MAX_VELOCITY
#     dth2 = observation[5] * MAX_VELOCITY
#     near_goal = cos_th1 < -0.95
#     slow = abs(dth1) < 1.0 and abs(dth2) < 1.0
#     return bool(near_goal and slow)


# def noisy_reset_func():
#     # reset casuale su tutto lo spazio degli stati
#     th1  = np.random.uniform(-np.pi, np.pi)
#     th2  = np.random.uniform(-np.pi, np.pi)
#     dth1 = np.random.uniform(-1.0, 1.0)
#     dth2 = np.random.uniform(-1.0, 1.0)
#     state = np.array([th1, th2, dth1, dth2])
#     return dynamics_func.normalize_state(state)


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

###PAPER

# reward_func     = make_reward_func(S_lqr=S_lqr, rho=rho)
# terminated_func = make_terminated_func(S_lqr=S_lqr, rho=rho)
# terminated_func_eval = make_terminated_func(S_lqr=S_lqr, rho=rho)

# noisy_reset_func = make_noisy_reset_func(dynamics_func, terminated_func)
# def eval_reset_func():
#     terminated_func_eval.reset()
#     p1 = np.random.uniform(-0.1, 0.1)
#     p2 = np.random.uniform(-0.1, 0.1)
#     state = np.array([p1, p2, 0.0, 0.0])
#     return dynamics_func.normalize_state(state)

### V3


reward_func          = make_reward_func()
terminated_func      = make_terminated_func()
terminated_func_eval = make_terminated_func()

noisy_reset_func     = make_noisy_reset_func(dynamics_func)

def eval_reset_func():
    terminated_func_eval.reset()
    p1 = np.random.uniform(-np.pi + 0.3, np.pi - 0.8)
    p2 = np.random.uniform(-0.1, 0.1)
    v1, v2 = 0.0, 0.0
    return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))


#noisy_reset_func = make_noisy_reset_func(dynamics_func)
# _base_reset_func = make_noisy_reset_func(dynamics_func)

# def noisy_reset_func():
#     """Resetta il contatore N_HOLD ad ogni nuovo episodio."""
#     terminated_func.reset()
#     return _base_reset_func()

# def eval_reset_func():
#     p1 = 0.0 + np.random.uniform(-0.1, 0.1)   # parte dal basso
#     p2 = 0.0 + np.random.uniform(-0.1, 0.1)
#     v1 = 0.0
#     v2 = 0.0
#     state = np.array([p1, p2, v1, v2])
#     return dynamics_func.normalize_state(state)




env_kwargs = dict(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=MAX_STEPS,
)

train_env = make_vec_env(CustomEnv, n_envs=N_ENVS, env_kwargs=env_kwargs, seed=SEED)

eval_env = CustomEnv(**{
    **env_kwargs,
    "reset_func":      eval_reset_func,
    "terminated_func": terminated_func_eval,   # ← istanza eval separata
})

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS E LOGGING
# ══════════════════════════════════════════════════════════════════════════════

# ── path completo per questa run ─────────────────────────────────────
LOG_DIR = os.path.join(LOG_DIR_BASE, RUN_NAME)   # ← unica riga da aggiungere
os.makedirs(LOG_DIR, exist_ok=True)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=max(EVAL_FREQ // N_ENVS, 1),  # scala con n_envs
    n_eval_episodes=N_EVAL_EPISODES,
    verbose=1,
)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE SAC
# ══════════════════════════════════════════════════════════════════════════════

# agent = SAC(
#     MlpPolicy,
#     train_env,
#     policy_kwargs=dict(net_arch=NET_ARCH, n_critics=N_CRITICS),
#     learning_rate=LEARNING_RATE,
#     gamma=GAMMA,
#     tau=TAU,
#     buffer_size=BUFFER_SIZE,
#     batch_size=BATCH_SIZE,
#     ent_coef=ENT_COEF,
#     seed=SEED,
#     verbose=1,
#     tensorboard_log=os.path.join(LOG_DIR, "tb_logs"),
# )

# fisso target entropia

agent = SAC(
    MlpPolicy,
    train_env,
    policy_kwargs=dict(net_arch=NET_ARCH, n_critics=N_CRITICS),
    learning_rate=LEARNING_RATE,
    learning_starts= LEARNING_STARTS,
    gamma=GAMMA,
    tau=TAU,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    ent_coef=ENT_COEF,
    target_entropy = TARGET_ENTROPY,
    seed=SEED,
    verbose=1,
    tensorboard_log=os.path.join(LOG_DIR, "tb_logs"),
)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"RUN: {RUN_NAME}")
    print(f"Obs dim: {OBS_DIM} | Episode: {EP_SECONDS:.1f}s ({MAX_STEPS} steps)")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,} | Buffer: {BUFFER_SIZE:,}")
    print(f"Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | Seed: {SEED}")
    print(f"{'='*60}\n")

    monitor = TrainingMonitorCallback(
        log_path=os.path.join(LOG_DIR, "training_monitor.csv"),
        dynamics_func=dynamics_func,
        L1=mpar.l[0],
        L2=mpar.l[1],
        S_lqr=S_lqr,    # ← aggiungi
        rho=rho,        # ← aggiungi
        max_velocity=MAX_VELOCITY,
        verbose=1,
    )



    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback,monitor],
        progress_bar=True,
    )
    agent.save(os.path.join(LOG_DIR, "sac_pendubot_final"))
