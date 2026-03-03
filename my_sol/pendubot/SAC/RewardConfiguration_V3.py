import numpy as np
from config import (
    MAX_VELOCITY, L1, L2,
    GOAL_POS_THRESHOLD, GOAL_VEL_THRESHOLD,
    N_HOLD, PROB_FAR, RESET_VEL,
)

# ══════════════════════════════════════════════════════
# PARAMETRI
# ══════════════════════════════════════════════════════

# Peso penalità quadratica errore posizione (solo vicino al goal)
W_POS   = 10.0
# Peso penalità velocità (attivo sempre, cresce vicino al goal)
W_VEL   = 1.0
# Peso bonus altezza end-effector (swing-up)
W_H     = 5.0
# Peso penalità azione
W_ACT   = 1e-3
# Raggio di "vicinanza al goal" — dentro questo raggio W_POS e W_VEL scalano
GOAL_RADIUS = 0.3   # rad — distanza angolare da π

W_ANGLE = 3.0

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _decode_obs(obs):
    p1 = np.arctan2(obs[1], obs[0])
    p2 = np.arctan2(obs[3], obs[2])
    v1 = obs[4] * MAX_VELOCITY
    v2 = obs[5] * MAX_VELOCITY
    return p1, p2, v1, v2

def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _end_effector_height(p1, p2):
    # massimo = L1+L2 quando p1=p2=π (tutto su)
    return -L1 * np.cos(p1) - L2 * np.cos(p1 + p2)

def _goal_proximity(err_q1, err_q2):
    """
    Scalare in [0, 1]:
    0 = lontano dal goal
    1 = esattamente al goal
    Transizione smooth con una gaussiana sull'errore di posizione.
    """
    dist = np.sqrt(err_q1**2 + err_q2**2)
    return np.exp(-dist**2 / (2 * GOAL_RADIUS**2))

def _in_goal(p1, p2, v1, v2):
    err_q1 = _wrap(p1 - np.pi)
    err_q2 = _wrap(p2)
    return (abs(err_q1) < GOAL_POS_THRESHOLD and
            abs(err_q2) < GOAL_POS_THRESHOLD and
            abs(v1)     < GOAL_VEL_THRESHOLD  and
            abs(v2)     < GOAL_VEL_THRESHOLD)

# ══════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════

def make_reward_func():
    def reward_func(observation, action):
        p1, p2, v1, v2 = _decode_obs(observation)
        u = float(action[0])

        err_q1 = _wrap(p1 - np.pi)
        err_q2 = _wrap(p2)

        # proximity ∈ [0,1]: quanto siamo vicini al goal
        prox = _goal_proximity(err_q1, err_q2)

        # ── Swing-up: bonus altezza normalizzata ──────────────────────────
        h      = _end_effector_height(p1, p2)
        h_max  = L1 + L2
        r_h    = W_H * (h / h_max)   # ∈ [-W_H, +W_H]

        # aggiungi: bonus esplicito per p1 vicino a π
        cos_p1 = np.cos(p1)   # = -1 al goal
        r_goal_dir = 2.0 * (cos_p1 + 1.0) / 2.0   # ∈ [0, 2], max a p1=π
        # oppure più aggressivo:
        r_goal_dir = 3.0 * cos_p1   # massimo +3 a π, minimo -3 a 0

        r_angle = W_ANGLE * (-np.cos(p1) - 1.0)


        # ── Stabilizzazione: penalità posizione, scala con proximity ──────
        r_pos  = -W_POS * prox * (err_q1**2 + err_q2**2)

        # ── Velocità: penalità che cresce vicino al goal ───────────────────
        vel_norm = (v1**2 + v2**2) / (MAX_VELOCITY**2)
        r_vel  = -W_VEL * (1.0 + 4.0 * prox) * vel_norm
        #                   ↑ lontano: peso 1x  |  al goal: peso 5x

        # ── Azione ────────────────────────────────────────────────────────
        r_act  = -W_ACT * u**2

        return float(r_h +r_angle+r_goal_dir+ r_pos + r_vel + r_act)

    return reward_func

# ══════════════════════════════════════════════════════
# TERMINATED
# ══════════════════════════════════════════════════════

def make_terminated_func():
    state = {"hold_count": 0}

    def terminated_func(observation):
        p1, p2, v1, v2 = _decode_obs(observation)
        if _in_goal(p1, p2, v1, v2):
            state["hold_count"] += 1
        else:
            state["hold_count"] = 0
        return state["hold_count"] >= N_HOLD

    def reset():
        state["hold_count"] = 0

    terminated_func.reset = reset
    return terminated_func

# ══════════════════════════════════════════════════════
# RESET
# ══════════════════════════════════════════════════════

def make_noisy_reset_func(dynamics_func):
    def noisy_reset_func():
        r = np.random.rand()
        if r < PROB_FAR:
            p1 = np.random.uniform(-np.pi + 0.5, np.pi - 1.0)
        else:
            p1 = np.pi + np.random.uniform(-0.4, 0.4)
        p2 = np.random.uniform(-0.3, 0.3)
        v1 = np.random.uniform(-RESET_VEL, RESET_VEL)
        v2 = np.random.uniform(-RESET_VEL, RESET_VEL)
        return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))
    return noisy_reset_func
