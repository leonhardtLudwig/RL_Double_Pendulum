import os
from datetime import datetime

from linearize_double_pendulum import linearize_double_pendulum


import matplotlib.pyplot as plt

import numpy as np


from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.pid.point_pid_controller import PointPIDController

from double_pendulum.analysis.leaderboard import get_swingup_time

def condition1(t, x):
    return False

def condition2(t, x):
    return False

model_par_path = 'acrobot_parameters.yml'
mpar = model_parameters(filepath=model_par_path)

active_act = 1
torque_limit = mpar.tl


stable_eq = [0.0, 0.0, 0.0, 0.0]
unstable_eq = [np.pi, 0.0, 0.0, 0.0]

dt = 0.001
t_final = 10.0
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

print("--- ANALISI EQUILIBRIO STABILE ---")
A_stable, B_stable = linearize_double_pendulum(mpar, q1d=stable_eq[0], q2d=stable_eq[1])
print("Matrice A:\n", np.round(A_stable, 2))
print("Matrice B:\n", np.round(B_stable, 2))

controller1 = PointPIDController(torque_limit=mpar.tl, dt=dt)
controller1.set_parameters(Kp=5, Kd=1.0, Ki=0.0)

print("\n--- ANALISI EQUILIBRIO INSTABILE ---")
A_unstable, B_unstable = linearize_double_pendulum(mpar, q1d=unstable_eq[0], q2d=unstable_eq[1])
print("Matrice A:\n", np.round(A_unstable, 2))
print("Matrice B:\n", np.round(B_unstable, 2))

Q = np.diag([1, 1, 0.1, 0.1])
R = np.diag([1, 1])*0.1
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(unstable_eq)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15)


def condition2(t, x):
    angle_error = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    vel_norm = np.linalg.norm(x[2:])
    return abs(angle_error) < 0.05 and vel_norm < 0.1


def condition1(t, x):
    angle_error = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    vel_norm = np.linalg.norm(x[2:])
    return abs(angle_error) > 0.15 and vel_norm > 0.1

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)
controller.init()

controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
    video_name="acrobot.mp4",
)

plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    save_to="ts_acrobot.png",
    show=False,
)
