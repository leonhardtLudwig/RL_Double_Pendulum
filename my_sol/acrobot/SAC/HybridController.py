import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.model.model_parameters import model_parameters
from stable_baselines3 import SAC
from config import GOAL, Q_LQR, R_LQR_MAT


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class HybridController(AbstractController):
    def __init__(self, sac_model_path, dynamics_func, S_lqr, rho,
                 model_par_path, dt):
        AbstractController.__init__(self)
        self.dynamics_func = dynamics_func
        self.S_lqr = np.asarray(S_lqr)
        self.rho = rho
        self.dt = dt
        self.active = "sac"
        self._prev_active = "sac"
        self._switch_log = []

        # ── SAC ──────────────────────────────────────────────────────────────
        self.sac = SAC.load(sac_model_path)
        obs_dim = self.sac.observation_space.shape[0]
        self.sac.predict(np.zeros(obs_dim))      # warmup

        # ── LQR ──────────────────────────────────────────────────────────────
        mpar = model_parameters(filepath=model_par_path)
        self.lqr = LQRController(model_pars=mpar)
        self.lqr.set_goal(GOAL)
        self.lqr.set_cost_matrices(Q=Q_LQR, R=R_LQR_MAT)
        self.lqr.set_parameters(failure_value=0.0, cost_to_go_cut=20.0)
        self.lqr.init()

    # def _near_goal(self, x):
    #     err_q1 = _wrap(x[0] - np.pi)
    #     err_q2 = _wrap(x[1])
    #     return (abs(err_q1) < 0.15 and abs(err_q2) < 0.15 and
    #             abs(x[2]) < 1.0 and abs(x[3]) < 1.0)
    def _near_goal(self, x):
        delta = np.array([_wrap(x[0] - np.pi), _wrap(x[1]), x[2], x[3]])
        return float(np.einsum("i,ij,j", delta, self.S_lqr, delta)) < self.rho

    def get_control_output_(self, x, t=None):
        if self._near_goal(x):
            self.active = "lqr"
            x_w = x.copy()
            x_w[0] = _wrap(x[0])
            x_w[1] = _wrap(x[1])
            u = self.lqr.get_control_output(x_w, t)
        else:
            self.active = "sac"
            obs = self.dynamics_func.normalize_state(x)
            action, _ = self.sac.predict(obs, deterministic=True)
            u = self.dynamics_func.unscale_action(action)

        # log switch
        if self.active != self._prev_active:
            t_now = t if t is not None else -1
            print(f"  [t={t_now:.3f}s] SWITCH: {self._prev_active} → {self.active}")
            self._switch_log.append((t_now, self._prev_active, self.active))
            self._prev_active = self.active

        return u

    def print_switch_summary(self):
        if not self._switch_log:
            print("Nessuno switch — ha usato solo SAC.")
            return
        print(f"\n{'─'*40}")
        print(f"Switch totali: {len(self._switch_log)}")
        for t, frm, to in self._switch_log:
            print(f"  t={t:.3f}s  {frm} → {to}")
        print(f"Controller finale: {self.active}")
        print(f"{'─'*40}")
