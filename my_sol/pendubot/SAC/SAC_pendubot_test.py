import os
import numpy as np
from gymnasium import spaces

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from stable_baselines3 import SAC

# sostituisci i valori hardcoded con quelli da config
from config import RUN_NAME, DT, INTEGRATOR, STATE_REPR, MAX_VELOCITY, LOG_DIR_BASE

from HybridController import HybridController
from TripleController import TripleController 
from ComputeRoa import compute_and_save_roa

# poi usa
model_path = os.path.join(LOG_DIR_BASE, RUN_NAME, "best_model", "best_model")
integrator = INTEGRATOR


import RewardLogger

# ── parametri modello ──────────────────────────────────────────────────────────
model_par_path = "pendubot_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

goal = [np.pi, 0.0, 0.0, 0.0]



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
dt = 0.01
integrator = "runge_kutta"
state_representation = 3
max_velocity = 20.0

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


# ── istanza controller ─────────────────────────────────────────────────────────
# controller = FixedSACController(
#     model_path="./log_data/SAC_pendubot/" + RUN_NAME + "/best_model/best_model",
#     dynamics_func=dynamics_func,
#     dt=dt,
# )
# controller.init()

#S_lqr, rho = compute_and_save_roa(model_par_path=model_par_path, verbose=False)
# ROA_S_PATH   = os.path.join("./", "roa_S.npy")
# ROA_RHO_PATH = os.path.join("./", "roa_rho.npy")

# if os.path.exists(ROA_S_PATH) and os.path.exists(ROA_RHO_PATH):
#     print("RoA già calcolata, carico da file...")
#     S_lqr = np.load(ROA_S_PATH)
#     rho   = float(np.load(ROA_RHO_PATH)[0])
# else:
#     print("Calcolo RoA...")
#     S_lqr, rho = compute_and_save_roa(
#         model_par_path=model_par_path,
#         save_S_path=ROA_S_PATH,
#         save_rho_path=ROA_RHO_PATH,
#         verbose=True,
#     )

# Q_lqr_online = np.diag([1.92, 1.92, 0.3, 0.3])
# R_lqr_online = np.diag([0.82, 0.82])
# S_online = np.array([
#     [7.857934201124567153e01, 5.653751913776947191e01, 1.789996146741196981e01, 8.073612858295813766e00],
#     [5.653751913776947191e01, 4.362786774581156379e01, 1.306971194928728330e01, 6.041705515910111401e00],
#     [1.789996146741196981e01, 1.306971194928728330e01, 4.125964000971944046e00, 1.864116086667296113e00],
#     [8.073612858295813766e00, 6.041705515910111401e00, 1.864116086667296113e00, 8.609202333737846491e-01]
# ])
# rho_online = 8.690673829091186575e-01  # 1.690673829091186575e-01

# controller = HybridController(
#     sac_model_path="./log_data/SAC_pendubot/" + RUN_NAME + "/best_model/best_model",
#     dynamics_func=dynamics_func,
#     S_lqr=S_online, rho=rho_online,
#     model_par_path=model_par_path,
#     dt=dt)
# controller.init()


def condition2(t, x):
    # attiva LQR: vicino al goal, q2 allineato, velocità contenuta
    p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
    v1, v2 = x[2], x[3]
    return (abs(p1_err) < 0.6 and   # ~20° da ±π
            abs(p2)     <  0.6  and   # ~28° di q2
            abs(v1)     < 20.0  and   # velocità residua tollerata
            abs(v2)     < 20.0)

def condition1(t, x):
    # torna al SAC: LQR ha perso il controllo
    p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
    return (abs(p1_err) > 0.6 or    # troppo lontano da ±π
            abs(p2)     > 0.3)      # q2 completamente fuori


# SAC_controller = FixedSACController(
#      model_path="./log_data/SAC_pendubot/" + RUN_NAME + "/best_model/best_model",
#      dynamics_func=dynamics_func,
#      dt=dt,
# )

# Q_stable = np.diag([250.0, 30.0, 15, 15])  
# R_stable = np.diag([0.2, 0.2]) 
# LQR_Controller = LQRController(model_pars=mpar)
# LQR_Controller.set_goal(goal)
# LQR_Controller.set_cost_matrices(Q=Q_stable, R=R_stable)
# LQR_Controller.set_parameters(failure_value=0.0,
#                           cost_to_go_cut=1000)

# controller = CombinedController(
#     controller1=SAC_controller,
#     controller2=LQR_Controller,
#     condition1=condition1,
#     condition2=condition2,
#     compute_both=False,
#     verbose = True,
# )
_lqr_active    = [False]
_stable_active = [False]

def reset_latches():
    _lqr_active[0]    = False
    _stable_active[0] = False

# def condition_goal(t, x):
#     """LQR_fast → LQR_stable: velocità quasi nulle, vicino all'equilibrio."""
#     if _stable_active[0]:
#         return True           # latch: rimane in LQR_stable
#     if not _lqr_active[0]:
#         return False          # LQR_fast non ancora attivo
#     err_q1 = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
#     err_q2 = (x[1] + np.pi) % (2*np.pi) - np.pi
#     v1, v2 = x[2], x[3]
#     triggered = (abs(err_q1) < 0.2 and
#                  abs(err_q2) < 0.4 and
#                  abs(v1)     < 1.5 and
#                  abs(v2)     < 6.5)
#     if triggered:
#         print(f"[SWITCH → LQR_stable] t={t:.3f}s  x={x}")
#         _stable_active[0] = True
#     return triggered

# def condition2(t, x):
#     if _lqr_active[0]:
#         return True 
#     # attiva LQR: vicino al goal, q2 allineato, velocità contenuta
#     p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
#     p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
#     v1, v2 = x[2], x[3]
#     triggered = (abs(p1_err) < 0.4 and   # ~20° da ±π
#             abs(p2)     <  0.5  and   # ~28° di q2
#             abs(v1)     < 10.0  and   # velocità residua tollerata
#             abs(v2)     < 20.0) 
#     if triggered:
#         print(f"[SWITCH → LQR] t={t:.3f}s  x={x}")  
#         _lqr_active[0] = True
#     return triggered

# def condition1(t, x):
#     if _lqr_active[0]:
#         return False
#     # torna al SAC: LQR ha perso il controllo
#     p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
#     p2     = (x[1] + np.pi) % (2*np.pi) - np.pi
#     return (abs(p1_err) > 0.6 or    # troppo lontano da ±π
#             abs(p2)     > 0.3)      # q2 completamente fuori


SAC_controller = FixedSACController(
     model_path="./log_data/SAC_pendubot/" + RUN_NAME + "/best_model/best_model",
     dynamics_func=dynamics_func,
     dt=dt,
)

Q_stable = np.diag([250.0, 30.0, 15, 15])  
R_stable = np.diag([0.2, 0.2]) 
LQR_Controller = LQRController(model_pars=mpar)
LQR_Controller.set_goal(goal)
LQR_Controller.set_cost_matrices(Q=Q_stable, R=R_stable)
LQR_Controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1000)

# Q_fast = np.diag([300.0, 140.0, 60.0, 120.0])
# R_fast = np.diag([0.25, 0.25])
# LQR_fast = LQRController(model_pars=mpar)
# LQR_fast.set_goal(goal)
# LQR_fast.set_cost_matrices(Q=Q_fast, R=R_fast)
# LQR_fast.set_parameters(failure_value=0,
#                           cost_to_go_cut=1e9)


# Q_stable = np.diag([80.0, 200.0, 30.0, 200.0])
# R_stable = np.diag([1.0, 1.0])*0.5

# LQR_stable = LQRController(model_pars=mpar)
# LQR_stable.set_goal(goal)
# LQR_stable.set_cost_matrices(Q=Q_stable, R=R_stable)
# LQR_stable.set_parameters(failure_value=0.0, cost_to_go_cut=1e9)

controller = CombinedController(
    controller1=SAC_controller,
    controller2=LQR_Controller,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
    verbose = True,
)

# controller = TripleController(
#     controller1=SAC_controller,
#     controller2=LQR_fast,
#     controller3=LQR_stable,
#     condition1=condition1,
#     condition2=condition2,
#     condition_goal=condition_goal,
#     compute_both=False,
#     verbose=True,
# )

controller.init()

LQR_Controller.init()
print("K =", LQR_Controller.K)
print("S diag =", np.diag(LQR_Controller.S))

# LQR_fast.init()
# print("K =", LQR_fast.K)
# print("S diag =", np.diag(LQR_fast.S))


# ── simulazione ────────────────────────────────────────────────────────────────
T, X, U = simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=2.0,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
    video_name="sac_pendubot.mp4",
    scale=0.3,
)

# ── Incolla qui i tuoi T, X, U dopo la simulazione ───────────────────────────
# Esempio d'uso:
#   T, X, U = simulator.simulate_and_animate(...)
#   inspect_window(T, X, U)

def inspect_window(T, X, U, t_start=0.66, t_end=0.70, dt=0.01):
    T = np.array(T)
    X = np.array(X)
    U = np.array(U)

    # U ha len(T)-1 elementi — allinea
    T = T[:len(U)]
    X = X[:len(U)]

    mask = (T >= t_start) & (T <= t_end)
    T_w  = T[mask]
    X_w  = X[mask]
    U_w  = U[mask]

    print(f"{'t':>6}  {'p1':>8}  {'p2':>8}  {'v1':>8}  {'v2':>8}  {'u1':>8}")
    print("-" * 60)
    for t, x, u in zip(T_w, X_w, U_w):
        p1_err = (x[0] - np.pi + np.pi) % (2*np.pi) - np.pi
        print(f"{t:6.3f}  {x[0]:8.4f}  {x[1]:8.4f}  {x[2]:8.4f}  {x[3]:8.4f}  {u[0]:8.4f}"
              f"   [err_p1={p1_err:+.3f}]")

    print()
    print("── STATS finestra ──────────────────────────────────────")
    print(f"v1  min={X_w[:,2].min():+.3f}  max={X_w[:,2].max():+.3f}")
    print(f"v2  min={X_w[:,3].min():+.3f}  max={X_w[:,3].max():+.3f}")
    print(f"u1  min={U_w[:,0].min():+.3f}  max={U_w[:,0].max():+.3f}")
    print(f"Stato a t={T_w[-1]:.3f}: {X_w[-1]}")

inspect_window(T, X, U, t_start=0.56, t_end=0.63, dt=dt)

# ── plot ───────────────────────────────────────────────────────────────────────
plot_timeseries(
    T, X, U,
    pos_y_lines=[np.pi],
    tau_y_lines=[-mpar.tl[0], mpar.tl[0]],
    save_to="ts_sac_pendubot.png",
    show=False,
)


X = np.array(X)   # aggiungi questa riga

idx = int(0.65 / dt)
x_h = X[idx].copy()
x_h[0] = (x_h[0] - np.pi + np.pi) % (2*np.pi) - np.pi
x_h[1] = (x_h[1] + np.pi) % (2*np.pi) - np.pi
#x_h -= LQR_fast.xd
x_h -=LQR_Controller.xd

#cost = float(x_h @ LQR_fast.S @ x_h)
cost = float(x_h @ LQR_Controller.S @ x_h)

print(f"v1={X[idx,2]:.1f}, v2={X[idx,3]:.1f}, cost-to-go={cost:.0f}")