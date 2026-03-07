import os
import numpy as np
from gymnasium import spaces

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.plotting import plot_timeseries
from stable_baselines3 import SAC

# sostituisci i valori hardcoded con quelli da config
from config import RUN_NAME, DT, INTEGRATOR, STATE_REPR, MAX_VELOCITY, LOG_DIR_BASE


# poi usa
model_path = os.path.join(LOG_DIR_BASE, RUN_NAME, "best_model", "best_model")
dt         = DT
integrator = INTEGRATOR

import RewardLogger

# ── parametri modello ──────────────────────────────────────────────────────────
model_par_path = "acrobot_parameters.yml"
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


# ── dynamics func ──────────────────────────────────────────────────────────────
integrator = "runge_kutta"
state_representation = 3
max_velocity = 20.0

dynamics_func = DynamicsFuncWrapper(
    double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot="acrobot",
        state_representation=state_representation,
        max_velocity=max_velocity,
        torque_limit=mpar.tl,
        scaling=True,
    )
)

# ── controller SAC corretto ────────────────────────────────────────────────────
class FixedSACController(SACController):
    def __init__(self, model_path, dynamics_func, dt, scaling=True):
        AbstractController.__init__(self)

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt
        self.scaling = scaling

        obs_dim = self.model.observation_space.shape[0]
        self.model.predict(np.zeros(obs_dim))

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.dynamics_func.unscale_action(action)


SAC_controller = FixedSACController(
     model_path="./log_data/SAC_acrobot/" + RUN_NAME + "/best_model/best_model",
     dynamics_func=dynamics_func,
     dt=dt,
)
SAC_controller.init()


# ── simulazione ────────────────────────────────────────────────────────────────
T, X, U = simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=10.0,
    dt=dt,
    controller=SAC_controller,
    integrator=integrator,
    save_video=True,
    video_name="sac_acrobot.mp4",
    scale=0.3,
)


# ── plot ───────────────────────────────────────────────────────────────────────
plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi],
    tau_y_lines=[-mpar.tl[0], mpar.tl[0]],
    save_to="ts_sac_acrobot.png",
    show=False,
)
