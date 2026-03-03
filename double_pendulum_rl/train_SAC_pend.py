import os
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import CustomEnv, double_pendulum_dynamics_func


# ── parametri modello ──────────────────────────────────────────────────────────
model_par_path = "pendubot_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)


# ── wrapper float32 ────────────────────────────────────────────────────────────
class DynamicsFuncWrapper:
    def __init__(self, dynamics_func):
        self.dynamics_func = dynamics_func
        self.max_velocity = dynamics_func.max_velocity

    def __call__(self, state, action, scaling=True):
        obs = self.dynamics_func(state, action, scaling=scaling)
        return obs.astype(np.float32)

    def normalize_state(self, state):
        return self.dynamics_func.normalize_state(state).astype(np.float32)

    def unscale_state(self, state):
        return self.dynamics_func.unscale_state(state)


# ── configurazione environment ─────────────────────────────────────────────────
dt = 0.01
integrator = "runge_kutta"
state_representation = 3
max_velocity = 20.0
max_steps = 1000

dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot="pendubot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=mpar.tl,
        scaling=True,
    )
)

obs_space = spaces.Box(np.array([-1.0]*6), np.array([1.0]*6), dtype=np.float32)
act_space = spaces.Box(np.array([-1.0]),   np.array([1.0]),   dtype=np.float32)


# ── reward / terminated / reset ────────────────────────────────────────────────
def reward_func(observation, action):
    th1  = np.arctan2(observation[1], observation[0])
    th2  = np.arctan2(observation[3], observation[2])
    dth1 = observation[4] * max_velocity
    dth2 = observation[5] * max_velocity

    err1 = np.abs(th1) - np.pi
    err2 = th2

    return (
        - 2.0  * err1**2
        - 1.0  * err2**2
        - 0.1  * (dth1**2 + dth2**2)
        - 0.01 * float(action[0])**2
    )


def terminated_func(observation):
    return False


def noisy_reset_func():
    th1 = np.random.uniform(-0.2, 0.2)
    th2 = np.random.uniform(-0.2, 0.2)
    state = np.array([th1, th2, 0.0, 0.0])
    return dynamics_func.normalize_state(state)


# ── environment ────────────────────────────────────────────────────────────────
env_kwargs = dict(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

train_env = make_vec_env(CustomEnv, n_envs=1, env_kwargs=env_kwargs)

eval_env = CustomEnv(**env_kwargs)
check_env(eval_env)


# ── callbacks ──────────────────────────────────────────────────────────────────
log_dir = "./log_data/SAC_pendubot"
os.makedirs(log_dir, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(log_dir, "best_model"),
    log_path=log_dir,
    eval_freq=10_000,
    n_eval_episodes=10,
    verbose=1,
)


# ── agente SAC ─────────────────────────────────────────────────────────────────
agent = SAC(
    MlpPolicy,
    train_env,
    policy_kwargs=dict(net_arch=[256, 256], n_critics=2),
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    buffer_size=1_000_000,
    batch_size=256,
    ent_coef="auto",
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
)


# ── training ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
    agent.save(os.path.join(log_dir, "sac_pendubot_final"))
