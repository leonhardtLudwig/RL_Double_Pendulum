import os
import numpy as np
import gymnasium as gym
import stable_baselines3
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from test_pend import PendulumPlant, pend_dynamics_func, plant_parameters

from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import CustomEnv

print('----------------------------')
print(stable_baselines3.version_file)
print('----------------------------')

def wrap_angles_diff(x):
    y = np.copy(x)
    y[0] = x[0] % (2*np.pi)
    while np.abs(y[0]) > np.pi:
        y[0] -= 2*np.pi
    return y

epsilon = 0.2


log_dir = "./log_data/SAC_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

dt = 0.01
integrator = "runge_kutta"
# integrator = "odeint"

robot = 'pendulum'

plant = PendulumPlant(**plant_parameters)
simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 3
obs_space = spaces.Box(np.array([-1.0]*state_representation), np.array([1.0]*state_representation), dtype=np.float32)
act_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
max_steps = 10000
############################################################################

#tuning parameter
n_envs = 1
training_steps = int(1e6) # default = 1e6
log_dir = "./log_data/SAC_training"
verbose = 1
# reward_threshold = -0.01
reward_threshold = 3e7
eval_freq=10000
n_eval_episodes=20
learning_rate=0.0003
##############################################################################
# initialize double pendulum dynamics
dynamics_func = pend_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=8.0,
    torque_limit=[plant_parameters['torque_limit']]
)


def reward_func(observation, action):
    # quadratic
    if state_representation == 2:
        s = np.array(
            [
                (observation[0] * np.pi + np.pi) % (2 * np.pi),  # [0, 2pi]
                observation[1] * dynamics_func.max_velocity,
            ]
        )
    elif state_representation == 3:
        s = np.array(
            [
                np.abs(np.arctan2(observation[0], observation[1])),
                observation[2] * dynamics_func.max_velocity,
            ]
        )

    # u = plant_parameters['torque_limit'] * action

    goal = [np.pi, 0.]

    # r = np.einsum("i, ij, j", s - goal, Q, s - goal) #+ np.einsum("i, ij, j", u, R, u)
    # r = np.clip(np.sqrt(r)/np.pi, 0, 1)
    r = 0.0
    if abs(s[0]) > np.pi / 2:
        r = +1.0
    if abs(s[0]) < 0.5:
        r = -1.0

    # r = 1.0 * np.exp(-((np.abs(s[0]) - goal[0]) / np.pi) **2) # attractor to pi
    # r += -1.0 * (np.exp(-(np.abs(s[0]) / np.pi) **2)) # repellor from 0


    # if observation[1] <= 0.5:
    #     r += 10
    #
    # if observation[1] <= 0.0:
    #     r += 10
    # # print('----------------')
    # print(observation)
    # print(s)
    # print(r)
    # print('----------------')

    return r

def terminated_func(observation):
    if state_representation == 2:
        s = np.array(
            [
                observation[0] * np.pi + np.pi,  # [0, 2pi]
                observation[1] * 8.0,
            ]
        )
    elif state_representation == 3:
        s = np.array(
            [
                np.arctan2(observation[0], observation[1]),
                observation[2] * 8.0,
            ]
        )
    else:
        raise NotImplementedError

    if np.abs(s[0] - np.pi) < epsilon and abs(s[1]) < 1.:
        return True
    else:
        return False


def noisy_reset_func():
    if state_representation == 2:
        rand = np.array(np.random.rand(state_representation) * 0.01, dtype=np.float32)
        observation = np.array([-1, 0.0], dtype=np.float32) + rand
    elif state_representation == 3:
        th = np.random.randn(1) * 0.1
        observation = np.clip(np.array([np.sin(th), np.cos(th), 0.0+np.random.rand(1)*0.01], dtype=np.float32), -1, 1)
    return observation.reshape(state_representation,)

def zero_reset_func():
    if state_representation == 2:
        observation = [-1.0, 0.0]
    else:
        observation = [0.0, 1.0, 0.0]
    return observation

# initialize vectorized environment
env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training env
envs = make_vec_env(
    env_id=CustomEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reward_func": reward_func,
        "terminated_func": terminated_func,
        "reset_func": noisy_reset_func,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
    },
)

# evaluation env
eval_env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training callbacks
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=os.path.join(log_dir,
                                      "best_model"),
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

# train
agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
    # learning_rate=learning_rate,
    gamma=0.99,
    action_noise=NormalActionNoise(mean=[0.0], sigma=[.1])
)

check_env(env)
if __name__ == '__main__':
    agent.learn(total_timesteps=training_steps, callback=eval_callback, progress_bar=False)
