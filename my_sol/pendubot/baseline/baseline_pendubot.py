import os
from datetime import datetime


import matplotlib.pyplot as plt
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.partial_feedback_linearization.pfl import EnergyShapingPFLAndLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.pid.point_pid_controller import PointPIDController

from ProgressiveSwingController import ProgressiveFeedbackSwingController
from TripleController import TripleController 




# Switch condition between Swing-up & LQR


def condition1(t, x):
    """Swings up when far from equilibria"""
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi
    return abs(err_q1) > 0.6 or abs(err_q2) > 0.6


def condition2(t, x):
    """Impose LQR when near to equilibria"""

    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi

    pos_err = 0.2
    vel_err = 2

    pos_ok = abs(err_q1) < pos_err and abs(err_q2) < pos_err
    vel_ok = abs(x[2]) < vel_err and abs(x[3]) < vel_err

    return pos_ok or vel_ok


def condition_goal(t, x):  # ← CAMBIA QUI: (t, x) come le altre
  
    """
    GOAL CONDITION: Active "lock" when equilibria position is reached and velocities are low.
    Once is true, it remains active only this controller
    """
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi
    
    pos_threshold = 0.1  
    vel_threshold = 0.3 
    
    goal_reached = (abs(err_q1) < pos_threshold and 
                    abs(err_q2) < pos_threshold and
                    abs(x[2]) < vel_threshold and 
                    abs(x[3]) < vel_threshold)
    
    return goal_reached



# Simulation setup

model_par_path = 'pendubot_parameters.yml'
mpar = model_parameters(filepath=model_par_path)

active_act = 0
torque_limit = mpar.tl

stable_eq = [0.0, 0.0, 0.0, 0.0]
unstable_eq = [np.pi, 0, 0.0, 0.0]

dt = 0.001
t_final = 20.0
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)


# Swing-up controller

controller1 = ProgressiveFeedbackSwingController(
    torque_limit=mpar.tl,
    kp=15.0,
    kd=1.85,
    torque_fraction=1.0
)


# LQR for oscillations

Q = np.diag([200.0, 40.0, 10.0, 10.0])
R = np.diag([5, 5])

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(unstable_eq)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=20)


# Stabilizing controller (robust LQR)

# Opzione 1: LQR con parametri più conservativi
Q_stable = np.diag([300.0, 60.0, 15.0, 15.0])*0.5  
R_stable = np.diag([3, 3]) 

controller3 = LQRController(model_pars=mpar)
controller3.set_goal(unstable_eq)
controller3.set_cost_matrices(Q=Q_stable, R=R_stable)
controller3.set_parameters(failure_value=0, cost_to_go_cut=20)

# Triple controller

controller = TripleController(
    controller1=controller1,
    controller2=controller2,
    controller3=controller3,
    condition1=condition1,
    condition2=condition2,
    condition_goal=condition_goal,
    compute_both=False,
    #verbose=True 
)

controller.init()

# Simulation

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0, 0, 0, 0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
    video_name="pendubot.mp4",
    scale=0.3
)

plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    save_to="ts_pendubot.png",
    show=False,
)

