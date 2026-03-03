
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# SOGLIE — derivate dalla baseline
# ══════════════════════════════════════════════════════════════════════════════

# da condition2 (bacino LQR)
LQR_POS_THRESHOLD   = 0.2
LQR_VEL_THRESHOLD   = 2.0

# da condition_goal (stabilizzazione)
GOAL_POS_THRESHOLD  = 0.1
GOAL_VEL_THRESHOLD  = 0.3

# pesi reward
W_POS       = 2.0     # peso errore posizione q1
W_POS2      = 0.5     # peso errore posizione q2
W_VEL       = 0.1     # peso penalità velocità
W_ACTION    = 0.05    # peso penalità azione
BONUS_LQR   = 5.0     # bonus dentro il bacino LQR
BONUS_GOAL  = 20.0    # bonus al raggiungimento del goal

# reset curriculum
PROB_FAR    = 0.7     # probabilità di reset lontano da π
RESET_VEL   = 1.0     # ampiezza velocità iniziale casuale [rad/s]

# necessario per de-normalizzare le velocità — deve matchare MAX_VELOCITY nel training
MAX_VELOCITY = 20.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _angular_errors(observation):
    """Restituisce (err_q1, err_q2, dth1, dth2) dallo stato normalizzato."""
    th1  = np.arctan2(observation[1], observation[0])
    th2  = np.arctan2(observation[3], observation[2])
    dth1 = observation[4] * MAX_VELOCITY
    dth2 = observation[5] * MAX_VELOCITY

    # stessa logica del wrap usato in condition1 / condition2 / condition_goal
    err_q1 = (th1 - np.pi + np.pi) % (2 * np.pi) - np.pi
    err_q2 = (th2 + np.pi) % (2 * np.pi) - np.pi

    return err_q1, err_q2, dth1, dth2


# ══════════════════════════════════════════════════════════════════════════════
# REWARD
# ══════════════════════════════════════════════════════════════════════════════

def reward_func(observation, action):
    err_q1, err_q2, dth1, dth2 = _angular_errors(observation)

    r = (
        - W_POS    * err_q1**2
        - W_POS2   * err_q2**2
        - W_VEL    * (dth1**2 + dth2**2) / MAX_VELOCITY**2
        - W_ACTION * float(action[0])**2
    )

    # bonus bacino LQR — da condition2
    in_lqr_basin = abs(err_q1) < LQR_POS_THRESHOLD and abs(err_q2) < LQR_POS_THRESHOLD
    if in_lqr_basin:
        r += BONUS_LQR

    # bonus goal — da condition_goal
    goal_reached = (
        abs(err_q1) < GOAL_POS_THRESHOLD and
        abs(err_q2) < GOAL_POS_THRESHOLD and
        abs(dth1)   < GOAL_VEL_THRESHOLD and
        abs(dth2)   < GOAL_VEL_THRESHOLD
    )
    if goal_reached:
        r += BONUS_GOAL

    return float(r)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINATED
# ══════════════════════════════════════════════════════════════════════════════

def terminated_func(observation):
    """Termina l'episodio con successo quando condition_goal è soddisfatta."""
    err_q1, err_q2, dth1, dth2 = _angular_errors(observation)

    return bool(
        abs(err_q1) < GOAL_POS_THRESHOLD and
        abs(err_q2) < GOAL_POS_THRESHOLD and
        abs(dth1)   < GOAL_VEL_THRESHOLD and
        abs(dth2)   < GOAL_VEL_THRESHOLD
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESET
# ══════════════════════════════════════════════════════════════════════════════

def make_noisy_reset_func(dynamics_func):
    """
    Restituisce la reset function con curriculum implicito.
    Riceve dynamics_func come argomento per evitare dipendenze circolari.
    """
    def noisy_reset_func():
        if np.random.rand() < PROB_FAR:
            # lontano da π — zona condition1 attiva (swing-up)
            th1 = np.random.uniform(-np.pi, np.pi - 0.6)
        else:
            # vicino a π — zona condition2 (stabilizzazione)
            th1 = np.pi + np.random.uniform(-0.4, 0.4)

        th2  = np.random.uniform(-0.3, 0.3)
        dth1 = np.random.uniform(-RESET_VEL, RESET_VEL)
        dth2 = np.random.uniform(-RESET_VEL, RESET_VEL)

        state = np.array([th1, th2, dth1, dth2])
        return dynamics_func.normalize_state(state)

    return noisy_reset_func
