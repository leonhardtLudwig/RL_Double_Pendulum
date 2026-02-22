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
from double_pendulum.controller.partial_feedback_linearization.pfl import EnergyShapingPFLAndLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.pid.point_pid_controller import PointPIDController

from ProgressiveSwingController import ProgressiveSwingController
from ProgressiveSwingController import ProgressiveFeedbackSwingController
from TripleController import TripleController  # ← AGGIUNGI QUESTO IMPORT

from double_pendulum.analysis.leaderboard import get_swingup_time


# ============================================================================
# CONDIZIONI DI SWITCH TRA SWING-UP E LQR
# ============================================================================

def condition1(t, x):
    """Passa a swing-up quando siamo lontani dall'equilibrio"""
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi
    return abs(err_q1) > 0.6 or abs(err_q2) > 0.6


def condition2(t, x):
    """Passa a LQR quando siamo vicini all'equilibrio"""
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi

    pos_err = 0.2
    vel_err = 2

    pos_ok = abs(err_q1) < pos_err and abs(err_q2) < pos_err
    vel_ok = abs(x[2]) < vel_err and abs(x[3]) < vel_err

    return pos_ok or vel_ok


def condition_goal(t, x):  # ← CAMBIA QUI: (t, x) come le altre
    """
    CONDIZIONE CRITICA: Attiva il "lock" quando raggiungiamo
    la posizione desiderata E le velocità sono basse.
    Una volta True, rimani bloccato sul terzo controllore.
    """
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi
    
    # Soglie RIGIDE per attivare la stabilizzazione finale
    pos_threshold = 0.1  # ±5.7° dal target
    vel_threshold = 0.3  # Velocità quasi nulla
    
    goal_reached = (abs(err_q1) < pos_threshold and 
                    abs(err_q2) < pos_threshold and
                    abs(x[2]) < vel_threshold and 
                    abs(x[3]) < vel_threshold)
    
    return goal_reached


# ============================================================================
# SETUP SIMULAZIONE
# ============================================================================

model_par_path = 'pendubot_parameters.yml'
mpar = model_parameters(filepath=model_par_path)

active_act = 0
torque_limit = mpar.tl

stable_eq = [0.0, 0.0, 0.0, 0.0]
unstable_eq = [np.pi, 0, 0.0, 0.0]

dt = 0.001
t_final = 60.0
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)


# ============================================================================
# CONTROLLER 1: SWING-UP
# ============================================================================

controller1 = ProgressiveFeedbackSwingController(
    torque_limit=mpar.tl,
    kp=15.0,
    kd=1.85,
    torque_fraction=1.0
)


# ============================================================================
# CONTROLLER 2: LQR (per mantenere la posizione durante le oscillazioni)
# ============================================================================

Q = np.diag([200.0, 40.0, 10.0, 10.0])
R = np.diag([5, 5])

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(unstable_eq)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=20)


# ============================================================================
# CONTROLLER 3: STABILIZZATORE FINALE (LQR ROBUSTO)
# ============================================================================
# Questo controllore entra in gioco SOLO quando condition_goal è True
# e rimane attivo per sempre (no switch)

# Opzione 1: LQR con parametri più conservativi
Q_stable = np.diag([300.0, 60.0, 15.0, 15.0])*0.5  # Più aggressivo sulla posizione
R_stable = np.diag([3, 3])  # Consuma più torque ma è più stabile

controller3 = LQRController(model_pars=mpar)
controller3.set_goal(unstable_eq)
controller3.set_cost_matrices(Q=Q_stable, R=R_stable)
controller3.set_parameters(failure_value=0, cost_to_go_cut=20)


# ============================================================================
# CONTROLLER TRIPLO
# ============================================================================

controller = TripleController(
    controller1=controller1,
    controller2=controller2,
    controller3=controller3,
    condition1=condition1,
    condition2=condition2,
    condition_goal=condition_goal,
    compute_both=False,
    verbose=True  # Stampa i cambi di controller
)

controller.init()

# ============================================================================
# SIMULAZIONE
# ============================================================================

print("\n" + "="*80)
print("INIZIO SIMULAZIONE CON TRIPLE CONTROLLER")
print("="*80 + "\n")

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0, 0, 0, 0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
    video_name="pendubot_triple.mp4",
    scale=0.3
)

plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    save_to="ts_pendubot_triple.png",
    show=False,
)

print("\n" + "="*80)
print("SIMULAZIONE COMPLETATA")
print("="*80)