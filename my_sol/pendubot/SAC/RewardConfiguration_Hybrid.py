import numpy as np
from config import MAX_VELOCITY, L1, L2, RESET_VEL, TRAIN_POS_THRESHOLD

S_ONLINE = np.array([
    [7.857934201124567153e+01, 5.653751913776947191e+01, 1.789996146741196981e+01, 8.073612858295813766e+00],
    [5.653751913776947191e+01, 4.362786774581156379e+01, 1.306971194928728330e+01, 6.041705515910111401e+00],
    [1.789996146741196981e+01, 1.306971194928728330e+01, 4.125964000971944046e+00, 1.864116086667296113e+00],
    [8.073612858295813766e+00, 6.041705515910111401e+00, 1.864116086667296113e+00, 8.609202333737846491e-01]
])
RHO_ONLINE = 1.690673829091186575e-01


# ── pesi ──────────────────────────────────────────────────────────────────────
W_H   = 20.0       # altezza end-effector
W_ACT = 5e-3      # penalità azione
W_VEL = 2.0       # penalità velocità (leggera: vogliamo energia per lo swing)
ROA_BONUS = 1000.0 # bonus terminale quando si entra nella RoA


Q_COST = np.diag([10.0, 10.0, 0.1, 0.1])
R_COST = np.array([[1e-4]])
GOAL   = np.array([np.pi, 0.0, 0.0, 0.0])
TORQUE = 10.0
W_ACT  = 5e-3
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

# def _in_roa(p1, p2, v1, v2, S_lqr, rho):
#     if S_lqr is None or rho is None:
#         return False
#     x = np.array([_wrap(p1 - np.pi), _wrap(p2), v1, v2])
#     return float(x @ S_lqr @ x) <= rho
def _in_roa(p1, p2, v1, v2):
    delta = np.array([_wrap(p1 - np.pi), _wrap(p2), v1, v2])
    return float(np.einsum("i,ij,j", delta, S_ONLINE, delta)) < 1.5*RHO_ONLINE

# ── reward ────────────────────────────────────────────────────────────────────
# def make_reward_func(S_lqr=None, rho=None):
#     def reward_func(observation, action):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         u = float(action[0])

#         err_q1 = _wrap(p1 - np.pi)
#         err_q2 = _wrap(p2)

#         # altezza normalizzata ∈ [-1, +1]
#         h = _end_effector_height(p1, p2)
#         r_h = W_H * (h / (L1 + L2))

#         # angolo: max=0 a p1=π, min=-2 a p1=0
#         r_angle = -np.cos(p1) - 1.0

#         # velocità (leggera)
#         r_vel = -W_VEL * (v1**2 + v2**2) / MAX_VELOCITY**2

#         # azione
#         r_act = -W_ACT * u**2

#         # bonus terminale: dato una sola volta grazie a terminated_func
#         #r_roa = ROA_BONUS if _in_roa(p1, p2, v1, v2, S_lqr, rho) else 0.0
#         # In RewardConfiguration, sostituisci il check RoA con:
#         r_near_goal = ROA_BONUS if (abs(err_q1) < 0.15 and abs(err_q2) < 0.15 and
#                     abs(v1) < 1.0 and abs(v2) < 1.0) else 0.0


#         return float(r_h + r_angle + r_vel + r_act + r_near_goal)

#     reward_func.reset = lambda: None   # interfaccia uniforme
#     return reward_func

# # ── terminated ────────────────────────────────────────────────────────────────
# def make_terminated_func(S_lqr=None, rho=None):
#     def terminated_func(observation):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         near_up = abs(_wrap(p1 - np.pi)) < TRAIN_POS_THRESHOLD   # 0.3 rad
#         near_q2 = abs(_wrap(p2))         < TRAIN_POS_THRESHOLD
#         return bool(near_up and near_q2)   # ← niente velocità
#     terminated_func.reset = lambda: None
#     return terminated_func


# # ── reset: sempre dal basso ───────────────────────────────────────────────────
# def make_noisy_reset_func(dynamics_func):
#     def noisy_reset_func():
#         r = np.random.rand()
#         if r < 0.3:
#             # curriculum: parte vicino al goal (p1=π)
#             p1 = np.pi + np.random.uniform(-0.4, 0.4)
#             p2 = np.random.uniform(-0.3, 0.3)
#             v1 = np.random.uniform(-0.5, 0.5)
#             v2 = np.random.uniform(-0.5, 0.5)
#         else:
#             # esplorazione normale dal basso
#             p1 = np.random.uniform(-0.5, 0.5)
#             p2 = np.random.uniform(-0.3, 0.3)
#             v1 = np.random.uniform(-RESET_VEL, RESET_VEL)
#             v2 = np.random.uniform(-RESET_VEL, RESET_VEL)
#         return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))
#     return noisy_reset_func

# def make_reward_func():
#     def reward_func(observation, action):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         u = np.array([TORQUE * float(action[0])])

#         delta = np.array([_wrap(p1 - GOAL[0]), _wrap(p2), v1, v2])
#         y1, y2 = _end_effector_height(p1, p2)

#         # costo quadratico sempre attivo
#         reward = -(np.einsum("i,ij,j", delta, Q_COST, delta) +
#                    np.einsum("i,ij,j", u, R_COST, u))

#         # bonus gerarchico: solo quando siamo in alto
#         if y1 > 0.8 * L1 and y2 > y1:
#             reward += (y2 / (L1 + L2)) * 1e3
#             velocity_penalty = max(0.0, abs(v1) + abs(v2) - 6.0)
#             reward -= velocity_penalty * 1e2
#             if _in_roa(p1, p2, v1, v2):
#                 reward += 1e4

#         return float(reward)

#     reward_func.reset = lambda: None
#     return reward_func

### BACKUP
# def make_reward_func():
#     def reward_func(observation, action):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         u = float(action[0])
#         y1, y2 = _end_effector_height(p1,p2)
#         h = y2
#         h_max = L1+L2

#         # segnale continuo verso l'alto
#         #r_height = -(np.cos(p1) + 1.0)          # 0 in alto, -2 in basso
#         h_normalized = (h/h_max +1)/2
#         r_height = 10*(np.exp(3*h_normalized)-1)
#         r_q2     = -0.1 * (1.0 - np.cos(p2))    # penalizza q2 lontano da 0 ORIGINALE


#         r_act    = -1e-3 * u**2

#         #r_energy = 0.1 * abs(v1)  # premiare la velocità angolare aiuta lo swing-up
#         r_energy = 0.1 * abs(v1)  # premiare la velocità angolare aiuta lo swing-up


#         # if y2 < 0.0:
#         #     r_vel = -0.02 *(v1**2+v2**2)/MAX_VELOCITY**2
#         # else:
#         #     r_vel = 0.0
#         r_vel = -0.5 *max(0,abs(v1)-10.0)/MAX_VELOCITY

#         if abs(v1) > 15.0:
#             r_height = -50.0
#         # bonus RoA solo quando ci sei
#         r_roa = 1e4 if _in_roa(p1, p2, v1, v2) else 0.0

#         return float(r_height + r_q2 + r_vel + r_act + r_roa+r_energy)

#     reward_func.reset = lambda: None
#     return reward_func


# def make_reward_func():
#     def reward_func(observation, action):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         u = float(action[0])
#         y1, y2 = _end_effector_height(p1,p2)
#         h = y2
#         h_max = L1+L2

#         # segnale continuo verso l'alto
#         #r_height = -(np.cos(p1) + 1.0)          # 0 in alto, -2 in basso
#         h_normalized = (h/h_max +1)/2
#         #r_height = 10*(np.exp(3*h_normalized)-1)
#         r_upright = 10.0 * (-np.cos(p1) + 1.0)   # 20 a q1=±π, 0 a q1=0

#         r_q2     = -0.1 * (1.0 - np.cos(p2))    # penalizza q2 lontano da 0 ORIGINALE


#         r_act    = -1e-3 * u**2

#         #r_energy = 0.1 * abs(v1)  # premiare la velocità angolare aiuta lo swing-up
#         phase = max(0.0, np.cos(p1))            # 1 in basso, 0 a 90°, -1 in alto
#         energy_weight = 0.1 + 0.5 * (1.0 - phase)   # più peso in alto
#         #r_energy = energy_weight * abs(v1)
#         r_energy = energy_weight * min(abs(v1), 6.0)   # cap a 8 rad/s, non premia accelerare oltre


#         # Distanza angolare da π, cala sempre avvicinandosi al goal
#         delta = np.pi - abs(p1)                 # 0 al goal, π al fondo
#         r_progress = -2.0 * delta              # segnale monotono, no falsi picchi


#         # if y2 < 0.0:
#         #     r_vel = -0.02 *(v1**2+v2**2)/MAX_VELOCITY**2
#         # else:
#         #     r_vel = 0.0
#         r_vel = -1*max(0,abs(v1)-8.0)/MAX_VELOCITY

#         # if abs(v1) > 15.0:
#         #     r_height = -50.0
#         # bonus RoA solo quando ci sei
#         r_clamp = -50.0 if abs(v1) > 12.0 else 0.0
#         r_roa = 1e4 if _in_roa(p1, p2, v1, v2) else 0.0

#         return float(r_clamp+r_q2 + r_vel + r_act + r_roa+r_energy+r_upright+r_progress)

#     reward_func.reset = lambda: None
#     return reward_func


####BACKUP DA COMBINARE CON LQR (QUASI FUNZIONANTE)
def make_reward_func():
    def reward_func(observation, action):
        p1, p2, v1, v2 = _decode_obs(observation)
        u = float(action[0])
        y1, y2 = _end_effector_height(p1,p2)
        h = y2
        h_max = L1+L2
        h_normalized = (h/h_max +1)/2

        dp1 = abs(abs(p1) - np.pi)   # 0 al goal, π al fondo
        dp2 = abs(p2)
        dv1 = abs(v1)
        dv2 = abs(v2)

        r_near_goal = 50.0 * np.exp(
            -(dp1**2 / 0.3**2 + dp2**2 / 0.5**2 + dv1**2 / 3.0**2 + dv2**2 / 3.0**2)
        )

        r_upright   = -10.0 * np.cos(p1)       # media zero su giro, max +10 a ±π

        r_vel = -0.5 * (v1**2 + v2**2) / MAX_VELOCITY**2

        r_base = -5.0 * (1.0 + np.cos(p1)) / 2.0

        r_energy = 2.0 * v1 * np.sign(p1)  # positivo se v1 e p1 concordano
        r_energy = np.clip(r_energy, -5, 5) # cap per evitare exploit

        near_top = max(0.0, (abs(p1) - 1.5) / (np.pi - 1.5))  # 0 lontano, 1 a ±π
        r_brake  = -3.0 * near_top * v1**2 / MAX_VELOCITY**2
        near_top = max(0.0, (abs(p1) - 1.5) / (np.pi - 1.5))
        r_q2     = -3.0 * near_top * (1.0 - np.cos(p2))



        reward = r_near_goal+r_upright+r_vel+r_base+r_energy+r_brake+r_q2


        return float(reward)

    reward_func.reset = lambda: None
    return reward_func


### REWARD CHE MI PERMETTE DI ARRIVARE FINO IN ALTO

# def make_reward_func():
#     def reward_func(observation, action):
#         p1, p2, v1, v2 = _decode_obs(observation)
#         u = float(action[0])
#         y1, y2 = _end_effector_height(p1,p2)
#         h = y2
#         h_max = L1+L2
#         h_normalized = (h/h_max +1)/2

#         dp1 = abs(abs(p1) - np.pi)   # 0 al goal, π al fondo
#         dp2 = abs(p2)
#         dv1 = abs(v1)
#         dv2 = abs(v2)

#         r_near_goal = 50.0 * np.exp(
#             -(dp1**2 / 0.3**2 + dp2**2 / 0.5**2 + dv1**2 / 3.0**2 + dv2**2 / 3.0**2)
#         )

#         r_upright   = -10.0 * np.cos(p1)       # media zero su giro, max +10 a ±π

#         r_vel = -0.5 * (v1**2 + v2**2) / MAX_VELOCITY**2

#         r_base = -5.0 * (1.0 + np.cos(p1)) / 2.0

#         r_energy = 2.0 * v1 * np.sign(p1)  # positivo se v1 e p1 concordano
#         r_energy = np.clip(r_energy, -5, 5) # cap per evitare exploit

#         near_top = max(0.0, (abs(p1) - 1.5) / (np.pi - 1.5))  # 0 lontano, 1 a ±π
#         r_brake  = -3.0 * near_top * v1**2 / MAX_VELOCITY**2
#         near_top = max(0.0, (abs(p1) - 1.5) / (np.pi - 1.5))
#         r_q2     = -3.0 * near_top * (1.0 - np.cos(p2))
#         r_v2_vel = -2.0 * near_top * v2**2 / MAX_VELOCITY**2  # ← aggiunta
#         r_act = -0.01 * (u / 10)**2


#         reward = r_near_goal+r_upright+r_vel+r_base+r_energy+r_brake+r_q2+r_v2_vel+r_act


#         return float(reward)

#     reward_func.reset = lambda: None
#     return reward_func






# ── terminated ────────────────────────────────────────────────────────────────
def make_terminated_func():
    def terminated_func(observation):
        p1, p2, v1, v2 = _decode_obs(observation)
        # if abs(v1) > 15.0 or abs(v2) > 15.0:
        #     return True

        return _in_roa(p1, p2, v1, v2)
    terminated_func.reset = lambda: None
    return terminated_func


# ── reset ─────────────────────────────────────────────────────────────────────

###BACKUP NOISY_RESET_MIGLIORE
def make_noisy_reset_func(dynamics_func):
    def noisy_reset_func():
        mode = np.random.choice(['bottom', 'mid', 'near_top'], p=[0.5, 0.3, 0.2])


        if mode == 'bottom':
            # reset standard dal basso
            p1 = np.random.uniform(-0.1, 0.1)
            p2 = np.random.uniform(-0.1, 0.1)
            v1 = np.random.uniform(-0.2, 0.2)
            v2 = np.random.uniform(-0.2, 0.2)

        elif mode == 'mid':
            # già a metà strada, con velocità nella direzione giusta
            sign = np.random.choice([-1, 1])
            p1 = sign * np.random.uniform(0.8, 2.0)
            p2 = np.random.uniform(-0.3, 0.3)
            v1 = sign * np.random.uniform(1.0, 4.0)  # spinge verso ±π
            v2 = np.random.uniform(-1.0, 1.0)

        else:  # near_top
            # vicino al goal, con velocità bassa
            sign = np.random.choice([-1, 1])
            p1 = sign * np.random.uniform(2.0, np.pi - 0.1)
            p2 = np.random.uniform(-0.3, 0.3)
            v1 = np.random.uniform(-2.0, 2.0)
            v2 = np.random.uniform(-1.0, 1.0)

        return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))
    return noisy_reset_func




### RESET CHE MI PERMETTE DI ARRIVARE FINO IN ALTO
# def make_noisy_reset_func(dynamics_func):
#     def noisy_reset_func():
#         mode = np.random.choice(
#                                 ['bottom', 'mid', 'near_top_slow', 'near_top_fast'],
#                                 p=[0.4, 0.25, 0.15, 0.20]
#                                 )

#         if mode == 'bottom':
#             # reset standard dal basso
#             p1 = np.random.uniform(-0.1, 0.1)
#             p2 = np.random.uniform(-0.1, 0.1)
#             v1 = np.random.uniform(-0.2, 0.2)
#             v2 = np.random.uniform(-0.2, 0.2)

#         elif mode == 'mid':
#             # già a metà strada, con velocità nella direzione giusta
#             sign = np.random.choice([-1, 1])
#             p1 = sign * np.random.uniform(0.8, 2.0)
#             p2 = np.random.uniform(-0.3, 0.3)
#             v1 = sign * np.random.uniform(1.0, 4.0)
#             v2 = np.random.uniform(-1.0, 1.0)

#         elif mode == 'near_top_slow':
#             # vicino al goal, con velocità bassa (come prima)
#             sign = np.random.choice([-1, 1])
#             p1 = sign * np.random.uniform(2.0, np.pi - 0.1)
#             p2 = np.random.uniform(-0.3, 0.3)
#             v1 = np.random.uniform(-2.0, 2.0)
#             v2 = np.random.uniform(-5.0, 5.0)

#         else:  # near_top_fast
#             # Stato difficile realistico: vicino alla zona finale ma con tanta velocità.
#             sign = np.random.choice([-1, 1])
#             p1 = sign * np.random.uniform(1.8, 2.6)   # ~103°–149°
#             p2 = np.random.uniform(-0.4, 0.4)
#             v1 = sign * np.random.uniform(6.0, 12.0)  # velocità tipica di arrivo
#             v2 = np.random.uniform(-8.0, 8.0)
#         return dynamics_func.normalize_state(np.array([p1, p2, v1, v2]))
#     return noisy_reset_func
