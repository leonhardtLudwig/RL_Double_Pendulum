import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from double_pendulum.controller.combined_controller import CombinedController
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import train_SAC_pend
from test_pend import PendulumPlant, pend_dynamics_func, plant_parameters, PController

from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import CustomEnv
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries


dt = 0.01
integrator = "runge_kutta"

robot = 'pendulum'

print(plant_parameters)

plant = PendulumPlant(**plant_parameters)
simulator = Simulator(plant=plant)
goal = [np.pi, 0.0]
state_representation = 3


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

controller1 = SACController(
    model_path = 'log_data/SAC_training/best_model/best_model.zip',
    dynamics_func=dynamics_func,
    dt=dt)

controller1.init()

# print(controller1.get_control_output_([0.]*2))
# exit()


# # initialize lqr controller

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=PController(dof=1, P=10),
    condition1=lambda t, x: abs(x[0]) < 2.,
    condition2=lambda t, x: abs(x[0]) >= 2.8,
    compute_both=False,
    verbose=True,
    dof=1
)
controller.init()

# start simulation
T, X, U, _ = simulator.rollout(x0=[0.0, 0.0],
                               T=10,
                               dt=dt,
                               policy=controller,
                               noise=None,
                               animate=False)

r = [train_SAC_pend.reward_func([np.sin(X[i,0]), np.cos(X[i, 0]), X[i, 1]], U[i]) for i in range(len(X))]



# print(X)
# print(U)

plt.subplot(2, 1, 1)
plt.plot(X[:, 0])
plt.plot(U[:, 0])
plt.plot([np.pi]*X.shape[0], color='r')
plt.plot([-np.pi]*X.shape[0], color='r')

plt.subplot(2, 1, 2)
plt.plot(r)
plt.show()

# # plot timeseries
# plot_timeseries(
#     T,
#     X,
#     U
# )
