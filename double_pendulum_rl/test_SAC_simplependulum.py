import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from test_pend import plant_parameters

from simple_pendulum.simulation.simulation import Simulator
from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.controllers.abstract_controller import AbstractController
from simple_pendulum.controllers.combined_controller import CombinedController
from train_SAC_simplependulum import SimplePendulumEnv


class SacController(AbstractController):
    """
    Controller which acts on a policy which has been learned with sac.
    """
    def __init__(self,
                 model_path,
                 torque_limit,
                 use_symmetry=True,
                 state_representation=2,
                 deterministic_control=False):
        """
        Controller which acts on a policy which has been learned with sac.

        Parameters
        ----------
        model_path : string
            path to the trained model in zip format
        torque_limit : float
            torque limit of the pendulum. The output of the model will be
            scaled with this number
        use_symmetry : bool
            whether to use the left/right symmetry of the pendulum
        """
        self.model = SAC.load(model_path)
        self.torque_limit = float(torque_limit)
        self.use_symmetry = bool(use_symmetry)
        self.state_representation = state_representation

        if state_representation == 2:
            # state is [th, th, vel]
            self.low = np.array([-6*2*np.pi, -20])
            self.high = np.array([6*2*np.pi, 20])
        elif state_representation == 3:
            # state is [cos(th), sin(th), vel]
            self.low = np.array([-1., -1., -8.])
            self.high = np.array([1., 1., 8.])
        self.deterministic_control = deterministic_control

    def get_control_output(self, meas_pos, meas_vel, meas_tau=0, meas_time=0):
        """
        The function to compute the control input for the pendulum actuator

        Parameters
        ----------
        meas_pos : float
            the position of the pendulum [rad]
        meas_vel : float
            the velocity of the pendulum [rad/s]
        meas_tau : float
            the meastured torque of the pendulum [Nm]
            (not used)
        meas_time : float
            the collapsed time [s]
            (not used)

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
            (not used, returns None)
        des_vel : float
            the desired velocity of the pendulum [rad/s]
            (not used, returns None)
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))

        # map meas pos to [-np.pi, np.pi]
        meas_pos_mod = np.mod(pos + np.pi, 2 * np.pi) - np.pi
        # observation = np.squeeze(np.array([meas_pos_mod, vel]))
        observation = self.get_observation([meas_pos_mod, vel])

        if self.use_symmetry:
            observation[0] *= np.sign(meas_pos_mod)
            observation[1] *= np.sign(meas_pos_mod)
            des_tau, _states = self.model.predict(observation, deterministic=self.deterministic_control)
            des_tau *= np.sign(meas_pos_mod)
        else:
            des_tau, _states = self.model.predict(observation, deterministic=self.deterministic_control)
        des_tau *= self.torque_limit

        # since this is a pure torque controller,
        # set pos_des and vel_des to None
        des_pos = None
        des_vel = None

        if np.abs(meas_pos) < 1e-4 and np.abs(meas_vel) < 1e-4:
            des_tau = self.torque_limit

        return des_pos, des_vel, float(des_tau)

    def get_observation(self, state):
        st = np.copy(state)
        st[1] = np.clip(st[1], self.low[-1], self.high[-1])
        if self.state_representation == 2:
            observation = np.array([obs for obs in st], dtype=np.float32)
        elif self.state_representation == 3:
            observation = np.array([np.cos(st[0]),
                                    np.sin(st[0]),
                                    st[1]],
                                   dtype=np.float32)

        return observation


class PController(AbstractController):
    def __init__(self, P=10.):
        self.P = P
        super().__init__()

    def get_control_output(self, meas_pos, meas_vel, meas_tau, meas_time):
        if meas_pos < 0:
            meas_pos += 2 * np.pi

        return None, None, float(self.P * (np.pi - meas_pos))



dt = 0.01
integrator = "runge_kutta"

robot = 'pendulum'

print(plant_parameters)

plant = PendulumPlant(**plant_parameters)
simulator = Simulator(plant=plant)
goal = [np.pi, 0.0]
state_representation = 3


eval_env = SimplePendulumEnv(simulator=simulator,
                             dt=dt,
                             max_steps=0,
                             reward_type='ternary',
                             integrator=integrator,
                             state_representation=state_representation)

# initialize double pendulum dynamics

controller1 = SacController(
    model_path='log_data/SAC_training/best_model/best_model.zip',
    # model_path='log_data/SAC_training/best_model/best_model_acquario.zip',
    torque_limit=2.25,
    state_representation=3,
    deterministic_control=True
)



# # initialize lqr controller

# initialize combined controller
# controller = controller1
controller = CombinedController(
    controller1=controller1,
    controller2=PController(P=10),
    condition1=lambda theta, theta_dot, tau, t: abs(theta) < 2.,
    condition2=lambda theta, theta_dot, tau, t: abs(theta) >= 2.8,
    compute_both=False)

# start simulation
T, X, U = simulator.simulate_and_animate(t0=0.0, x0=[0.0, 0.0], tf=10., dt=dt, controller=controller,
                                            integrator=integrator)
# print(X)
# print(U)

r = [eval_env.swingup_reward(X[i], U[i]) for i in range(len(X))]

X = np.array(X)
U = np.array(U)
T = np.array(T)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(X[:, 0])
plt.plot(U)
plt.plot([np.pi]*X.shape[0], color='r')
plt.plot([-np.pi]*X.shape[0], color='r')

plt.subplot(2,1, 2)
plt.plot(r)
plt.show()

# # plot timeseries
# plot_timeseries(
#     T,
#     X,
#     U
# )
