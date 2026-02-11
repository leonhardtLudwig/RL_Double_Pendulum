import os
from datetime import datetime

import matplotlib.pyplot
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

model_par_path = 'pendubot_parameters.yml'
torque_limit = [10.0, 0.0]
active_act = 0

mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

dt = 0.001
t_final = 5.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller1 = PointPIDController(torque_limit=torque_limit, dt=dt)
controller1.set_parameters(Kp=5, Kd=1.0, Ki=0.0)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
Q = 3.0 * np.diag([0.5, 0.5, 0.1, 0.1])
R = np.eye(2) * 0.1
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15)

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

print('Swingup time: '+str(get_swingup_time(T, np.array(X), mpar=mpar)))


# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
)
