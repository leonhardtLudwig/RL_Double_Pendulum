# TrainingMonitor.py

import numpy as np
import csv
import os
from stable_baselines3.common.callbacks import BaseCallback
#from RewardConfiguration_Paper import _in_lqr_roa
from RewardConfiguration_V3 import _in_goal as _in_lqr_roa




class TrainingMonitorCallback(BaseCallback):
    """
    Logga ad ogni episodio:
    - altezza massima raggiunta dall'end-effector
    - altezza media durante l'episodio
    - velocità massima raggiunta
    - se ha raggiunto la zona goal
    - return episodio
    """

    def __init__(self, log_path, dynamics_func, L1, L2, max_velocity,
                 S_lqr=None, rho=None,          # ← aggiungi
                 goal_pos_thr=0.2, goal_vel_thr=2.0, verbose=0):
        super().__init__(verbose)
        self.log_path       = log_path
        self.dynamics_func  = dynamics_func
        self.L1             = L1
        self.L2             = L2
        self.max_velocity   = max_velocity
        self.goal_pos_thr   = goal_pos_thr
        self.goal_vel_thr   = goal_vel_thr

        self.S_lqr = S_lqr
        self.rho   = rho

        # stato episodio corrente
        self._ep_reward     = 0.0
        self._ep_h_max      = -999.0
        self._ep_h_sum      = 0.0
        self._ep_v_max      = 0.0
        self._ep_steps      = 0
        self._ep_goal       = False

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "episode_return",
                "h_max", "h_mean", "v_max",
                "goal_reached", "ep_length"
            ])

    def _end_effector_height(self, p1, p2):
        return -self.L1 * np.cos(p1) - self.L2 * np.cos(p1 + p2)

    def _on_step(self) -> bool:

        


        obs    = self.locals["new_obs"][0]     # obs dopo lo step
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]

        # decodifica osservazione (state_repr=3)
        p1 = np.arctan2(obs[1], obs[0])
        p2 = np.arctan2(obs[3], obs[2])
        v1 = obs[4] * self.max_velocity
        v2 = obs[5] * self.max_velocity

        h = self._end_effector_height(p1, p2)
        v_tot = np.sqrt(v1**2 + v2**2)

        # debug temporaneo — rimuovi dopo
        # if h > 0.48:
        #     val = np.array([
        #         (p1 - np.pi + np.pi) % (2*np.pi) - np.pi,
        #         p2, v1, v2
        #     ])
        #     xSx = float(val @ self.S_lqr @ val) if self.S_lqr is not None else -1
        #     print(f"  h={h:.3f} | p1={p1:.3f} | v1={v1:.3f} | v2={v2:.3f} | xSx={xSx:.4f} | rho={self.rho}")

        # accumula statistiche episodio
        self._ep_reward += reward
        self._ep_h_max   = max(self._ep_h_max, h)
        self._ep_h_sum  += h
        self._ep_v_max   = max(self._ep_v_max, v_tot)
        self._ep_steps  += 1

        # controlla goal
        err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
        err_q2 = (p2 + np.pi) % (2*np.pi) - np.pi
        # if _in_lqr_roa(p1, p2, v1, v2, self.S_lqr, self.rho):
        #     self._ep_goal = True
        if _in_lqr_roa(p1, p2, v1, v2):
            self._ep_goal = True
        if done:
            h_mean = self._ep_h_sum / max(self._ep_steps, 1)

            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps:>8d}] "
                    f"ret={self._ep_reward:>8.1f} | "
                    f"h_max={self._ep_h_max:.3f} | "
                    f"h_mean={h_mean:.3f} | "
                    f"v_max={self._ep_v_max:.1f} | "
                    f"goal={'YES' if self._ep_goal else 'no':>3s} | "
                    f"steps={self._ep_steps}"
                )

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    round(self._ep_reward, 3),
                    round(self._ep_h_max, 4),
                    round(h_mean, 4),
                    round(self._ep_v_max, 3),
                    int(self._ep_goal),
                    self._ep_steps
                ])

            # reset
            self._ep_reward = 0.0
            self._ep_h_max  = -999.0
            self._ep_h_sum  = 0.0
            self._ep_v_max  = 0.0
            self._ep_steps  = 0
            self._ep_goal   = False

        return True
