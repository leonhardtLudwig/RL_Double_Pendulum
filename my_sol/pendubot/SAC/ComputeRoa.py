import numpy as np
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.roa.ellipsoid import sampleFromEllipsoid

from config import (
    MODEL_PAR_PATH, GOAL, Q_LQR, R_LQR_MAT,
    GOAL_POS_THRESHOLD, GOAL_VEL_THRESHOLD,
    ROA_N_SAMPLES, ROA_MAXITER, ROA_T_SIM,
    ROA_SUCCESS_RATIO, ROA_RHO_MAX, ROA_S_PATH, ROA_RHO_PATH,
    DT, INTEGRATOR,
)


def simulate_lqr_success(
    x0_abs,
    controller,
    plant,
    dt=DT,
    t_sim=ROA_T_SIM,
    integrator=INTEGRATOR,
    pos_threshold=GOAL_POS_THRESHOLD,
    vel_threshold=GOAL_VEL_THRESHOLD,   # ← allineato con reward config
    goal=GOAL,
):
    sim = Simulator(plant=plant)
    sim.set_state(0.0, x0_abs)
    sim.reset_data_recorder()

    n_steps = int(t_sim / dt)
    for _ in range(n_steps):
        # wrapping angolare prima di passare lo stato al LQR
        x_wrapped = sim.x.copy()
        x_wrapped[0] = (sim.x[0] + np.pi) % (2 * np.pi) - np.pi
        x_wrapped[1] = (sim.x[1] + np.pi) % (2 * np.pi) - np.pi
        u = controller.get_control_output(x=x_wrapped, t=sim.t)
        sim.step(u, dt, integrator=integrator)

    x_final = sim.x
    err_q1 = (x_final[0] - goal[0] + np.pi) % (2 * np.pi) - np.pi
    err_q2 = (x_final[1] - goal[1] + np.pi) % (2 * np.pi) - np.pi

    pos_ok = abs(err_q1) < pos_threshold and abs(err_q2) < pos_threshold
    vel_ok = abs(x_final[2]) < vel_threshold and abs(x_final[3]) < vel_threshold
    return pos_ok and vel_ok


def estimate_roa(
    controller,
    plant,
    S,
    goal=GOAL,
    rho_min=1e-6,
    rho_max=ROA_RHO_MAX,
    n_samples=ROA_N_SAMPLES,
    maxiter=ROA_MAXITER,
    dt=DT,
    t_sim=ROA_T_SIM,
    success_ratio=ROA_SUCCESS_RATIO,
    verbose=True,
):
    for i in range(maxiter):
        rho_probe = rho_min + (rho_max - rho_min) / 2.0

        successes = 0
        for _ in range(n_samples):
            x_bar = sampleFromEllipsoid(S, rho_probe, rInner=0.8, rOuter=1.0)
            x0_abs = goal + x_bar
            x0_abs[0] = (x0_abs[0] + np.pi) % (2 * np.pi) - np.pi  # forza in [-π, π]
            x0_abs[1] = (x0_abs[1] + np.pi) % (2 * np.pi) - np.pi

            if simulate_lqr_success(
                x0_abs=x0_abs,
                controller=controller,
                plant=plant,
                dt=dt,
                t_sim=t_sim,
            ):
                successes += 1

        ratio = successes / n_samples
        accepted = ratio >= success_ratio

        if verbose:
            print(
                f"  iter {i+1:2d} | rho_probe={rho_probe:.4f} | "
                f"success={successes}/{n_samples} ({ratio:.0%}) | "
                f"{'ACCETTATO ✓' if accepted else 'RIFIUTATO ✗'}"
            )

        if accepted:
            rho_min = rho_probe
        else:
            rho_max = rho_probe

    rho_final = rho_min
    if verbose:
        print(f"\n>>> rho stimato = {rho_final:.6f}")
    return rho_final


def compute_and_save_roa(
    model_par_path=MODEL_PAR_PATH,
    save_S_path=ROA_S_PATH,
    save_rho_path=ROA_RHO_PATH,
    verbose=True,
):
    mpar = model_parameters(filepath=model_par_path)
    plant = SymbolicDoublePendulum(model_pars=mpar)

    controller = LQRController(model_pars=mpar)
    controller.set_goal(GOAL)
    controller.set_cost_matrices(Q=Q_LQR, R=R_LQR_MAT)
    controller.set_parameters(failure_value=0.0, cost_to_go_cut=1000.0)
    controller.init()

    S = np.asarray(controller.S)
    K = np.asarray(controller.K)

    if verbose:
        print(f"S calcolata:\n{np.round(S, 3)}")
        print(f"K calcolata:\n{np.round(K, 3)}")
        print("\nAvvio stima RoA (bisezione + campionamento)...")

    rho = estimate_roa(
        controller=controller,
        plant=plant,
        S=S,
        verbose=verbose,
    )

    np.save(save_S_path, S)
    np.save(save_rho_path, np.array([rho]))
    if verbose:
        print(f"\nSalvati: {save_S_path}, {save_rho_path}")

    return S, rho


if __name__ == "__main__":
    compute_and_save_roa(verbose=True)
