# # reward_config.py
# # Reward a tre stadi — basata su Zhang, Sathuluri, Zimmermann (TU Munich, 2023)
# # https://arxiv.org/abs/2312.11311

# import numpy as np


# # ══════════════════════════════════════════════════════════════════════════════
# # PARAMETRI FISICI
# # ══════════════════════════════════════════════════════════════════════════════

# MAX_VELOCITY    = 20.0      # vmax [rad/s] — per normalizzazione
# L1              = 0.4       # lunghezza link 1 [m] — adatta ai tuoi mpar
# L2              = 0.1       # lunghezza link 2 [m] — adatta ai tuoi mpar


# # ══════════════════════════════════════════════════════════════════════════════
# # PARAMETRI REWARD — pendubot (Tabella I del paper)
# # ══════════════════════════════════════════════════════════════════════════════

# # Stadio 1 — reward quadratica sull'errore di stato
# Q1 = 150.0        # peso errore posizione q1
# Q2 = 5.0        # peso errore posizione q2
# Q3 = 0.1        # peso errore velocità q1
# Q4 = 0.1        # peso errore velocità q2



# R  = 1e-4       # peso azione
# # R = 5

# # Stadio 2 — bonus altezza end-effector
# H_LINE  = 0.8 * (L1 + L2)  # soglia altezza: 80% dell'altezza massima
# R_LINE  = 500.0               # bonus fisso quando end-effector supera h_line
# V_THRESH = 15.0              # soglia velocità [rad/s] — penalità spin
# R_VEL   = 0.0               # penalità velocità (solo acrobot nel paper, 0 per pendubot)

# #   attiva solo quando h >= H_LINE — spinge SAC a frenare vicino a π
# R_VEL_NEAR = 15.0            # peso penalità velocità near-goal

# #R_VEL_NEAR = 0.5

# # Stadio 3 — bonus RoA LQR
# # SLQR e rho vengono dal calcolo della RoA del tuo LQR — vedi sotto
# R_LQR   = 1e3               # bonus quando si è dentro la RoA

# # Soglie condition_goal — da baseline (usate in terminated_func)
# GOAL_POS_THRESHOLD = 0.2
# GOAL_VEL_THRESHOLD = 0.3

# #   evita che SAC impari a "sfiorare" π senza stabilizzarsi
# N_HOLD  = 10                # step consecutivi richiesti


# # Reset curriculum
# PROB_FAR    = 0.7
# RESET_VEL   = 1.0


# # ══════════════════════════════════════════════════════════════════════════════
# # HELPER
# # ══════════════════════════════════════════════════════════════════════════════

# def _decode_obs(observation):
#     """
#     Estrae (p1, p2, v1, v2) dall'osservazione normalizzata con state_repr=3.
#     obs = [cos(p1), sin(p1), cos(p2), sin(p2), v1/vmax, v2/vmax]
#     """
#     p1 = np.arctan2(observation[1], observation[0])   # [-π, π]
#     p2 = np.arctan2(observation[3], observation[2])   # [-π, π]
#     v1 = observation[4] * MAX_VELOCITY
#     v2 = observation[5] * MAX_VELOCITY
#     return p1, p2, v1, v2


# def _end_effector_height(p1, p2):
#     """
#     Altezza dell'end-effector rispetto al pivot — eq. (5) del paper.
#     h = -l1*cos(p1) - l2*cos(p1+p2)
#     Massimo = L1 + L2 quando p1=p2=π (entrambi su)
#     """
#     return -L1 * np.cos(p1) - L2 * np.cos(p1 + p2)


# def _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
#     if S_lqr is None:
#         err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
#         err_q2 = (p2 + np.pi) % (2*np.pi) - np.pi
#         return (abs(err_q1) < GOAL_POS_THRESHOLD and
#                 abs(err_q2) < GOAL_POS_THRESHOLD and
#                 abs(v1) < GOAL_VEL_THRESHOLD and
#                 abs(v2) < GOAL_VEL_THRESHOLD)
#     else:
#         # ← stessa normalizzazione del fallback, non p1 - π diretto
#         err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
#         x = np.array([err_q1, p2, v1, v2])
#         return float(x @ S_lqr @ x) <= rho


# # def _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
# #     """
# #     Controlla se lo stato è dentro la RoA del LQR.
# #     Se S_lqr è None usa fallback (distanza angolare).
# #     """
# #     if S_lqr is None:
# #         err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
# #         err_q2 = (p2 + np.pi) % (2*np.pi) - np.pi
# #         return (abs(err_q1) < GOAL_POS_THRESHOLD and
# #                 abs(err_q2) < GOAL_POS_THRESHOLD and
# #                 abs(v1) < GOAL_VEL_THRESHOLD and
# #                 abs(v2) < GOAL_VEL_THRESHOLD)
# #     else:
# #         x = np.array([p1 - np.pi, p2, v1, v2])
# #         return float(x @ S_lqr @ x) <= rho


# # ══════════════════════════════════════════════════════════════════════════════
# # REWARD — tre stadi (eq. 4 del paper)
# # ══════════════════════════════════════════════════════════════════════════════

# def make_reward_func(S_lqr=None, rho=None):
#     """
#     Factory che restituisce reward_func.
#     S_lqr: matrice 4x4 della RoA del LQR (np.ndarray), None = usa fallback
#     rho:   scalare soglia della RoA
#     """
#     def reward_func(observation, action):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         u = float(action[0])

#         # errori rispetto al goal [π, 0, 0, 0]
#         err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
#         err_q2 = (p2 + np.pi) % (2*np.pi) - np.pi
#         x_err = np.array([err_q1, err_q2, v1, v2])

#         # ── Stadio 1: reward quadratica ────────────────────────────────────────
#         Q_mat = np.diag([Q1, Q2, Q3, Q4])
#         r = -(x_err @ Q_mat @ x_err) - R * u**2

#         # ── Stadio 2: bonus altezza end-effector ───────────────────────────────
#         h = _end_effector_height(p1, p2)

#         h_max = L1 + L2
#         # bonus proporzionale all'altezza — sempre attivo, spinge verso l'alto
#         r += 2.0 * (h / h_max)   # in [-2, +2], +2 quando tutto su

#          #   quando l'end-effector è sopra h_line, SAC deve frenare
#         if h >= H_LINE:
#             r -= R_VEL_NEAR * (v1**2 + v2**2) / (MAX_VELOCITY**2)


#         # ── Stadio 3: bonus RoA LQR ────────────────────────────────────────────
#         if _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
#             r += R_LQR

#         return float(r)

#     return reward_func


# # ══════════════════════════════════════════════════════════════════════════════
# # TERMINATED — entra nella RoA → LQR prende il controllo
# # ══════════════════════════════════════════════════════════════════════════════

# # def make_terminated_func(S_lqr=None, rho=None):
# #     """
# #     Termina l'episodio solo dopo N_HOLD step consecutivi dentro la RoA.
# #     Evita che SAC impari a "toccare" π senza stabilizzarsi.
# #     Il contatore è mantenuto nella closure — si resetta a ogni nuovo episodio
# #     chiamando terminated_func.reset() (vedi gym_env.py o wrappa in un callback).
# #     """
# #     state = {"hold_count": 0}

# #     def terminated_func(observation):
# #         p1, p2, v1, v2 = _decode_obs(observation)
# #         if _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
# #             state["hold_count"] += 1
# #         else:
# #             state["hold_count"] = 0
# #         return state["hold_count"] >= N_HOLD

# #     def reset_counter():
# #         state["hold_count"] = 0

# #     terminated_func.reset = reset_counter
# #     return terminated_func

# def make_terminated_func(S_lqr=None, rho=None):
#     state = {"hold_count": 0, "n_episodes": 0}

#     def terminated_func(observation):
#         p1, p2, v1, v2 = _decode_obs(observation)

#         # N_HOLD cresce da 1 a N_HOLD con il curriculum
#         # ogni 500 episodi aumenta di 1
#         current_hold = min(1 + state["n_episodes"] // 20, N_HOLD)

#         if _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
#             state["hold_count"] += 1
#         else:
#             state["hold_count"] = 0

#         return state["hold_count"] >= current_hold

#     def reset_counter():
#         state["hold_count"] = 0
#         state["n_episodes"] += 1

#     terminated_func.reset = reset_counter
#     return terminated_func



# # ══════════════════════════════════════════════════════════════════════════════
# # RESET — curriculum 70/30
# # ══════════════════════════════════════════════════════════════════════════════

# # def make_noisy_reset_func(dynamics_func):
# #     def noisy_reset_func():
# #         if np.random.rand() < PROB_FAR:
# #             #p1 = np.random.uniform(-np.pi, np.pi - 0.6)   # lontano da π
# #             p1 = np.random.uniform(-np.pi + 0.3, np.pi - 0.8)   # escludi zone vicino a ±π

# #         else:
# #             p1 = np.pi + np.random.uniform(-0.4, 0.4)     # vicino a π

# #         p2   = np.random.uniform(-0.3, 0.3)
# #         v1   = np.random.uniform(-RESET_VEL, RESET_VEL)
# #         v2   = np.random.uniform(-RESET_VEL, RESET_VEL)
# #         state = np.array([p1, p2, v1, v2])
# #         return dynamics_func.normalize_state(state)

# #     return noisy_reset_func

# def make_noisy_reset_func(dynamics_func, terminated_func=None):
#     def noisy_reset_func():
#         # reset contatore — sempre, in ogni caso
#         if terminated_func is not None and hasattr(terminated_func, 'reset'):
#             terminated_func.reset()

#         if np.random.rand() < PROB_FAR:
#             p1 = np.random.uniform(-np.pi + 0.3, np.pi - 0.8)
#         else:
#             p1 = np.pi + np.random.uniform(-0.4, 0.4)

#         p2 = np.random.uniform(-0.3, 0.3)
#         v1 = np.random.uniform(-RESET_VEL, RESET_VEL)
#         v2 = np.random.uniform(-RESET_VEL, RESET_VEL)
#         state = np.array([p1, p2, v1, v2])
#         return dynamics_func.normalize_state(state)

#     return noisy_reset_func


# RewardConfiguration_Paper.py
# Reward a tre stadi — basata su Zhang, Sathuluri, Zimmermann (TU Munich, 2023)
# https://arxiv.org/abs/2312.11311

import numpy as np

# RewardConfiguration_Paper.py
from config import (
    MAX_VELOCITY, GOAL_POS_THRESHOLD, GOAL_VEL_THRESHOLD,
    Q1, Q2, Q3, Q4, R_ACTION, R_LQR, REWARD_SCALE,
    H_LINE_FRAC, R_VEL_NEAR, N_HOLD, PROB_FAR,PROB_NEAR, RESET_VEL
)
L1 = 0.4
L2 = 0.1
H_LINE = H_LINE_FRAC *(L1+L2)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI FISICI
# ══════════════════════════════════════════════════════════════════════════════

# MAX_VELOCITY = 20.0
# L1           = 0.4
# L2           = 0.1

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI REWARD
# ══════════════════════════════════════════════════════════════════════════════

# Q1 = 150.0
# Q2 = 5.0
# Q3 = 5.0
# Q4 = 5.0
# R  = 1e-3

# H_LINE     = 0.7 * (L1 + L2)
# R_VEL_NEAR = 2.0
# R_LQR      = 5000.0

# REWARD_SCALE = 1000.0

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI TERMINAZIONE
# ══════════════════════════════════════════════════════════════════════════════

# GOAL_POS_THRESHOLD = 0.2
# GOAL_VEL_THRESHOLD = 0.2
# N_HOLD             = 3

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI RESET
# ══════════════════════════════════════════════════════════════════════════════

# PROB_FAR  = 0.9
# PROB_NEAR = 0.1
# RESET_VEL = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _decode_obs(observation):
    p1 = np.arctan2(observation[1], observation[0])
    p2 = np.arctan2(observation[3], observation[2])
    v1 = observation[4] * MAX_VELOCITY
    v2 = observation[5] * MAX_VELOCITY
    return p1, p2, v1, v2

def _end_effector_height(p1, p2):
    return -L1 * np.cos(p1) - L2 * np.cos(p1 + p2)

def _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
    err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
    if S_lqr is None:
        err_q2 = (p2 + np.pi) % (2*np.pi) - np.pi
        return (abs(err_q1) < GOAL_POS_THRESHOLD and
                abs(err_q2) < GOAL_POS_THRESHOLD and
                abs(v1) < GOAL_VEL_THRESHOLD and
                abs(v2) < GOAL_VEL_THRESHOLD)
    else:
        x = np.array([err_q1, p2, v1, v2])
        return float(x @ S_lqr @ x) <= rho

# ══════════════════════════════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════════════════════════════

def make_reward_func(S_lqr=None, rho=None):
    def reward_func(observation, action):
        p1, p2, v1, v2 = _decode_obs(observation)
        u = float(action[0])

        err_q1 = (p1 - np.pi + np.pi) % (2*np.pi) - np.pi
        err_q2 = (p2 + np.pi) % (2*np.pi) - np.pi
        x_err  = np.array([err_q1, err_q2, v1, v2])

        # stadio 1 — penalità quadratica
        Q_mat = np.diag([Q1, Q2, Q3, Q4])
        r = -(x_err @ Q_mat @ x_err) - R_ACTION * u**2
        #r=0

        # stadio 2 — bonus altezza + freno spinning globale
        h     = _end_effector_height(p1, p2)
        h_max = L1 + L2
        r += 2.0 * (h / h_max)
        r -= 0.3 * (v1**2 + v2**2) / (MAX_VELOCITY**2)

        ## AAGIUNTA RUN 4 28/02
        # if abs(err_q1) > 0.5 and abs(v1) < 1.0:
        #     r -= 50.0   # non puoi stare fermo a metà strada

        if h >= H_LINE:
            r += R_VEL_NEAR * (v1**2 + v2**2) / (MAX_VELOCITY**2)

        # stadio 3 — bonus RoA
        if _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
            r += R_LQR

        return float(r / REWARD_SCALE)

    return reward_func

# ══════════════════════════════════════════════════════════════════════════════
# TERMINATED
# ══════════════════════════════════════════════════════════════════════════════

def make_terminated_func(S_lqr=None, rho=None):
    state = {"hold_count": 0, "n_episodes": 0}

    def terminated_func(observation):
        p1, p2, v1, v2 = _decode_obs(observation)
        current_hold = min(1 + state["n_episodes"] // 20, N_HOLD)

        

        if _in_lqr_roa(p1, p2, v1, v2, S_lqr, rho):
            print(f"GOAL → p1={p1:.3f} rad ({np.degrees(p1):.1f}°), v1={v1:.3f}")
            state["hold_count"] += 1
        else:
            state["hold_count"] = 0

        return state["hold_count"] >= current_hold

    def reset_counter():
        state["hold_count"] = 0
        state["n_episodes"] += 1

    terminated_func.reset = reset_counter
    return terminated_func

# ══════════════════════════════════════════════════════════════════════════════
# RESET
# ══════════════════════════════════════════════════════════════════════════════

def make_noisy_reset_func(dynamics_func, terminated_func=None):
    def noisy_reset_func():
        if terminated_func is not None and hasattr(terminated_func, 'reset'):
            terminated_func.reset()

        r = np.random.rand()          # un solo lancio
        if r < PROB_FAR:
            p1 = np.random.uniform(-np.pi + 0.3, np.pi - 0.8)
        else:
            p1 = np.pi + np.random.uniform(-0.4, 0.4)

        p2 = np.random.uniform(-0.3, 0.3)
        v1 = np.random.uniform(-RESET_VEL, RESET_VEL)
        v2 = np.random.uniform(-RESET_VEL, RESET_VEL)
        state = np.array([p1, p2, v1, v2])
        return dynamics_func.normalize_state(state)

    return noisy_reset_func

