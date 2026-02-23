import numpy as np

def linearize_double_pendulum(mpar, q1d, q2d):
    """
    Linearizzazione del doppio pendolo attorno a:
        q = (q1d, q2d)
        q_dot = 0
    Restituisce A, B per sistema completamente attuato (D = I)
    """

    # ---- Estrazione parametri corretta ----
    m1, m2 = mpar.m
    l1, l2 = mpar.l
    r1, r2 = mpar.r
    b1, b2 = mpar.b
    cf1, cf2 = mpar.cf
    I1, I2 = mpar.I
    Ir = mpar.Ir
    gr = mpar.gr
    g = mpar.g

    # ---- Pre-calcoli ----
    c1 = np.cos(q1d)
    c2 = np.cos(q2d)
    c12 = np.cos(q1d + q2d)

    # ---- Matrice di inerzia M(q_e) ----
    M11 = I1 + I2 + m2*l1**2 + 2*m2*l1*r2*c2 + gr**2*Ir + Ir
    M12 = I2 + m2*l1*r2*c2 - gr*Ir
    M22 = I2 + gr**2*Ir

    M = np.array([[M11, M12],
                  [M12, M22]])

    M_inv = np.linalg.inv(M)

    # ---- Jacobiano della gravità dG/dq ----
    dG1_dq1 = -g*m1*r1*c1 - g*m2*(l1*c1 + r2*c12)
    dG1_dq2 = -g*m2*r2*c12
    dG2_dq1 = -g*m2*r2*c12
    dG2_dq2 = -g*m2*r2*c12

    J_G = np.array([[dG1_dq1, dG1_dq2],
                    [dG2_dq1, dG2_dq2]])

    # ---- Jacobiano attrito in q_dot = 0 ----
    J_F = np.array([
        [b1 + 100*cf1, 0],
        [0, b2 + 100*cf2]
    ])

    # ---- Assemblaggio A ----
    A = np.zeros((4, 4))

    # q_dot = v
    A[0:2, 2:4] = np.eye(2)

    # ddot_q rispetto a q
    A[2:4, 0:2] = M_inv @ J_G

    # ddot_q rispetto a q_dot
    A[2:4, 2:4] = -M_inv @ J_F

    # ---- Matrice B (D = I) ----
    B = np.zeros((4, 2))
    B[2:4, :] = M_inv

    return A, B