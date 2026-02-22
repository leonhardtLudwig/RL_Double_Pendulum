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
# from double_pendulum.controller.energy.energy_Xin import EnergyController
# from double_pendulum.controller.inverse_dynamics.computed_torque_controller import ComputedTorqueController
from double_pendulum.controller.partial_feedback_linearization.pfl import EnergyShapingPFLAndLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.pid.point_pid_controller import PointPIDController

from ProgressiveSwingController import ProgressiveSwingController
from ProgressiveSwingController import ProgressiveFeedbackSwingController

from double_pendulum.analysis.leaderboard import get_swingup_time




def condition2(t, x):
    # errori wrappati rispetto a (π, 0)
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi

    pos_err = 0.2
    vel_err = 2

    pos_ok = abs(err_q1) < pos_err and abs(err_q2) < pos_err
    vel_ok = abs(x[2]) < vel_err and abs(x[3]) < vel_err

    return pos_ok or vel_ok


def condition1(t, x):
    err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi

    return abs(err_q1) > 0.6 or abs(err_q2) > 0.6

model_par_path = 'pendubot_parameters.yml'
mpar = model_parameters(filepath=model_par_path)

active_act = 0
torque_limit = mpar.tl


stable_eq = [0.0, 0.0, 0.0, 0.0]
unstable_eq = [np.pi, 0, 0.0, 0.0]

dt = 0.001
t_final =20.0
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)



Q = np.diag([200.0, 40.0, 10.0, 10.0])
R = np.diag([5, 5])

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(unstable_eq)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=20)

controller1 = ProgressiveFeedbackSwingController(torque_limit=mpar.tl)


controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
    verbose=True
)


controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    #x0=[np.pi-0.1, 0.0, 0.1, 0.5],
    x0 = [0,0,0,0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
    video_name="pendubot.mp4",
    scale=0.3  # <--- AGGIUNGI SOLO QUESTA RIGA
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

# Salva i dati della simulazione su file
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_file = f"simulation_data_{timestamp}.txt"

# with open(output_file, 'w') as f:
#     f.write("="*80 + "\n")
#     f.write("DATI SIMULAZIONE PENDUBOT\n")
#     f.write(f"Timestamp: {timestamp}\n")
#     f.write("="*80 + "\n\n")
    
#     # Dati dalla simulazione
#     f.write("Tempi (T):\n")
#     f.write(str(T) + "\n\n")
    
#     f.write("Stati (X) - [angle1, angle2, velocity1, velocity2]:\n")
#     f.write(str(X) + "\n\n")
    
#     f.write("Attuazioni (U) - [u1, u2]:\n")
#     f.write(str(U) + "\n\n")
    
#     f.write("-"*80 + "\n")
#     f.write("DATI ALTERNATIVI DAL SIMULATORE\n")
#     f.write("-"*80 + "\n\n")
    
#     f.write("t_values (Stessi tempi):\n")
#     f.write(str(sim.t_values) + "\n\n")
    
#     f.write("x_values (Stessi stati):\n")
#     f.write(str(sim.x_values) + "\n\n")
    
#     f.write("tau_values (Stesse attuazioni):\n")
#     f.write(str(sim.tau_values) + "\n\n")
    
#     f.write("meas_x_values (Stati misurati - se filtrati):\n")
#     f.write(str(sim.meas_x_values) + "\n\n")
    
#     f.write("con_u_values (Comandi del controllore):\n")
#     f.write(str(sim.con_u_values) + "\n\n")
    
#     f.write("="*80 + "\n")
#     f.write("INFORMAZIONI AGGIUNTIVE\n")
#     f.write("="*80 + "\n\n")
#     f.write(f"Durata simulazione: {t_final} s\n")
#     f.write(f"Timestep: {dt} s\n")
#     f.write(f"Numero passi: {len(T)}\n")
#     f.write(f"Tempo swing-up: {get_swingup_time(T, np.array(X), mpar=mpar)} s\n")

# print(f"\nDati simulazione salvati in: {output_file}")