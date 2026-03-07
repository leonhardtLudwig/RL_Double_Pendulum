import numpy as np
from config import MAX_VELOCITY, L1, L2

# ── helpers ───────────────────────────────────────────────────────────────────
def _decode_obs(obs):
    p1 = np.arctan2(obs[1], obs[0])
    p2 = np.arctan2(obs[3], obs[2])
    v1 = obs[4] * MAX_VELOCITY
    v2 = obs[5] * MAX_VELOCITY
    return p1, p2, v1, v2

def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _end_effector_height(p1, p2):
    y1 = -L1 * np.cos(p1)
    y2 = y1 - L2 * np.cos(p1 + p2)
    return y1, y2


def _in_goal(p1, p2, v1, v2):
    GOALPOSTHRESHOLD = 0.2
    GOALVELTHRESHOLD = 0.3
    dp1 = abs(abs(p1) - np.pi)   # 0 al goal, π al fondo
    dp2 = abs(p2)
    dv1 = abs(v1)
    dv2 = abs(v2)
    return (dp1 < GOALPOSTHRESHOLD and
            dp2 < GOALPOSTHRESHOLD and
            dv1 < GOALVELTHRESHOLD and
            dv2 < GOALVELTHRESHOLD)

# ── reward ────────────────────────────────────────────────────────────────────

def make_reward_func():
    def reward_func(observation, action):
        p1, p2, v1, v2 = _decode_obs(observation)
        u = float(action[0])   # per acrobot è u2
        y1, y2 = _end_effector_height(p1, p2)
        h = y2
        h_max = L1 + L2
        h_normalized = (h / h_max + 1) / 2

        dp1 = abs(abs(p1) - np.pi)
        dp2 = abs(p2)
        dv1 = abs(v1)
        dv2 = abs(v2)

        r_near_goal = 50.0 * np.exp(
            -(dp1**2 / 0.3**2 + dp2**2 / 0.5**2 + dv1**2 / 3.0**2 + dv2**2 / 3.0**2)
        )

        # acrobot: end-effector dipende da p1+p2, non solo p1
        r_upright = -8.0 * np.cos(p1) - 4.0 * np.cos(p1 + p2)

        r_vel = -0.5 * (v1**2 + v2**2) / MAX_VELOCITY**2

        r_base = -5.0 * (1.0 + np.cos(p1)) / 2.0

        # acrobot: pompa v2 (giunto attuato), non v1
        r_energy = 2.0 * v2 * np.sign(p1)
        r_energy = np.clip(r_energy, -5, 5)

        # near_top: usa altezza end-effector invece di solo p1
        near_top = max(0.0, (h_normalized - 0.7) / 0.3)  # 0 sotto 70% h_max, 1 al top
        r_brake  = -3.0 * near_top * v2**2 / MAX_VELOCITY**2   # frena v2 vicino al top
        r_q1     = -3.0 * near_top * (1.0 - np.cos(p1))        # penalizza q1 passivo fuori posto

        reward = r_near_goal + r_upright + r_vel + r_base + r_energy + r_brake + r_q1

        return float(reward)

    reward_func.reset = lambda: None
    return reward_func



# ── terminated ────────────────────────────────────────────────────────────────
def make_terminated_func():
    def terminated_func(observation):
        p1, p2, v1, v2 = _decode_obs(observation)
        return _in_goal(p1, p2, v1, v2)
    terminated_func.reset = lambda: None
    return terminated_func


# ── reset ─────────────────────────────────────────────────────────────────────

def make_noisy_reset_func(dynamics_func):
    def noisy_reset_func():
        mode = np.random.choice(['bottom', 'mid', 'near_top'], p=[0.5, 0.3, 0.2])

        if mode == 'bottom':
            p1 = np.random.uniform(-0.1, 0.1)
            p2 = np.random.uniform(-0.1, 0.1)
            v1 = np.random.uniform(-0.2, 0.2)
            v2 = np.random.uniform(-0.2, 0.2)

        elif mode == 'mid':
            sign = np.random.choice([-1, 1])
            p1 = sign * np.random.uniform(0.8, 2.0)
            p2 = np.random.uniform(-0.3, 0.3)
            v1 = np.random.uniform(-2.0, 2.0)
            v2 = sign * np.random.uniform(1.0, 4.0)   # v2 spinge (giunto attuato)

        else:  # near_top
            sign = np.random.choice([-1, 1])
            p1 = sign * np.random.uniform(2.0, np.pi - 0.1)
            p2 = np.random.uniform(-0.3, 0.3)
            v1 = np.random.uniform(-2.0, 2.0)
            v2 = np.random.uniform(-1.0, 1.0)

        return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))
    return noisy_reset_func

