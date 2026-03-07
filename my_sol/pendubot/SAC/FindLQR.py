import itertools
import numpy as np
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr import lqr as solve_lqr
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


# ── MODIFICA QUI ──────────────────────────────────────────────────────────────
MODEL_PAR_PATH = "pendubot_parameters.yml"
GOAL           = [np.pi, 0.0, 0.0, 0.0]
U_MAX          = 10.0

X_HANDOFF = np.array([3.29125593, 0.21077346, 0.76515986, 2.22417004])

Q1_VALS  = [50, 150, 250, 400, 800, 1500]
Q2_VALS  = [10, 30, 80, 150, 300]
QV1_VALS = [1, 5, 10, 30, 80]
QV2_VALS = [10, 50, 100, 200, 400]
R_VALS   = [0.02, 0.1, 0.5, 1.5, 5.0]
# ─────────────────────────────────────────────────────────────────────────────


mpar  = model_parameters(filepath=MODEL_PAR_PATH)
plant = SymbolicDoublePendulum(model_pars=mpar)

A = np.asarray(plant.linear_matrices(x0=np.array(GOAL), u0=[0.0, 0.0])[0])
B = np.asarray(plant.linear_matrices(x0=np.array(GOAL), u0=[0.0, 0.0])[1])

def wrap(x):
    e = x.copy().astype(float)
    e[0] = (e[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    e[1] = (e[1] + np.pi) % (2*np.pi) - np.pi
    return e

x_err = wrap(X_HANDOFF)

best  = {"score": np.inf}
total = len(Q1_VALS)*len(Q2_VALS)*len(QV1_VALS)*len(QV2_VALS)*len(R_VALS)
print(f"Testing {total} combinazioni...")
print("-" * 60)
LOG_EVERY = max(1, total // 20)

for i, (q1, q2, qv1, qv2, r) in enumerate(
    itertools.product(Q1_VALS, Q2_VALS, QV1_VALS, QV2_VALS, R_VALS), start=1
):
    Q = np.diag([q1, q2, qv1, qv2])
    R = np.diag([r, r])

    try:
        K, S, _ = solve_lqr(A, B, Q, R)
        K = np.asarray(K)
    except Exception:
        continue

    A_cl = A - B @ K
    eigs = np.linalg.eigvals(A_cl)

    # salta se instabile
    if np.any(np.real(eigs) >= 0):
        continue

    # velocità di convergenza: max real part degli autovalori (più negativo = meglio)
    max_eig = np.max(np.real(eigs))   # vogliamo minimizzare (più negativo = convergenza rapida)

    # torque iniziale comandato
    u_cmd = float((-K @ x_err).flatten()[0])
    u_clipped = np.clip(u_cmd, -U_MAX, U_MAX)

    # penalizza saturazione eccessiva e convergenza lenta
    saturation_penalty = max(0.0, abs(u_cmd) - U_MAX) * 10.0
    convergence_score  = -max_eig          # vogliamo max_eig molto negativo

    score = convergence_score + saturation_penalty

    if score < best["score"]:
        best = {"score": score, "Q": Q, "R": R, "K": K.copy(),
                "u_cmd": u_cmd, "max_eig": max_eig,
                "eigs": eigs.copy()}

    if i % LOG_EVERY == 0 or i == total:
        print(f"[{i:5d}/{total}]  {100*i/total:5.1f}%  |  "
              f"best convergence: {best['max_eig']:.3f}  "
              f"u_cmd: {best['u_cmd']:+.2f} Nm")

print()
print("=" * 60)
print("RISULTATO MIGLIORE")
print("=" * 60)
print(f"Q = np.diag({list(np.diag(best['Q']))})")
print(f"R = np.diag({list(np.diag(best['R']))})")
print(f"K = {best['K'].tolist()}")
print(f"Torque iniziale  : {best['u_cmd']:+.2f} Nm")
print(f"Autovalori CL    : {[f'{e.real:.2f}{e.imag:+.2f}j' for e in best['eigs']]}")
print(f"Max Re(eig)      : {best['max_eig']:.4f}  (più negativo = convergenza più rapida)")
print("=" * 60)
