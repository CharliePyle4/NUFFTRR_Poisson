import numpy as np

def compute_radial_integrals(r_m: np.ndarray,
                             f_fourier_coeff: np.ndarray,
                             quad_rule: int,
                             rad_unif: int):
    """
    Dispatch to the appropriate C_n, D_n radial integral routine.

    Parameters
    ----------
    r_m : ndarray, shape (M,)
        Radial grid.
    f_fourier_coeff : ndarray, shape (N+1, M)
        Fourier coefficients f_n(r_m).
    quad_rule : int
        Quadrature rule index (passed through to the underlying routines).
    rad_unif : int
        1 → uniform radial mesh, use compute_C_D_uniform
        0 → nonuniform radial mesh, use compute_C_D_nonuniform

    Returns
    -------
    C, D : ndarray
        Radial integral arrays used in the v^-, v^+ recurrences.
    """
    if rad_unif == 1:
        C, D = compute_C_D_uniform(r_m, f_fourier_coeff, quad_rule)
    elif rad_unif == 0:
        C, D = compute_C_D_nonuniform(r_m, f_fourier_coeff, quad_rule)
    else:
        raise ValueError('Incorrect index for "rad_unif"')
    return C, D


def compute_v_neg_pos(C: np.ndarray,
                      D: np.ndarray,
                      r_m: np.ndarray,
                      N: int,
                      M: int,
                      quad_rule: int):
    """
    Compute v^- and v^+ via radial recurrences.

    Parameters
    ----------
    C, D : ndarray, shape (N/2+1, M-1) or similar
        Radial integral arrays (as produced by compute_C_D_*).
    r_m : ndarray, shape (M,)
        Radial grid.
    N : int
        Number of angular points.
    M : int
        Number of radial points.
    quad_rule : int
        1 → trapezoidal 1-step recurrences
        2 → Simpson 2-step recurrences

    Returns
    -------
    v_neg, v_pos : ndarray, shape (N/2+1, M)
    """
    halfN = N // 2
    modes = np.arange(halfN + 1)        # 0..N/2
    n_mat = modes + 1                   # 1..N/2+1

    v_neg = np.zeros((halfN + 1, M), dtype=complex)
    v_pos = np.zeros((halfN + 1, M), dtype=complex)

    if quad_rule == 1:
        # Trapezoidal: 1‑step recurrences
        v_neg[:, 1] = C[:, 0]  # r_2

        exp_neg = modes - halfN
        for i in range(2, M):
            factor = (r_m[i] / r_m[i - 1]) ** exp_neg
            v_neg[:, i] = factor * v_neg[:, i - 1] + C[:, i - 1]

        exp_pos = modes
        for i in range(M - 2, -1, -1):
            factor = (r_m[i] / r_m[i + 1]) ** exp_pos
            v_pos[:, i] = factor * v_pos[:, i + 1] + D[:, i]

    elif quad_rule == 2:
        # Simpson: 2‑step recurrences
        exp_neg = n_mat - (halfN + 1)

        for i in range(2, M):
            ratio = r_m[i] / r_m[i - 2]
            factor = ratio ** exp_neg

            if i == 2:
                v_neg[:, 1] = C[:, 0]

            v_neg[:, i] = factor * v_neg[:, i - 2] + C[:, i - 1]

        exp_pos = n_mat - 1

        for i in range(M - 3, -1, -1):
            ratio = r_m[i] / r_m[i + 2]
            factor = ratio ** exp_pos

            if i == M - 3:
                v_pos[:, M - 2] = D[:, 0]

            v_pos[:, i] = factor * v_pos[:, i + 2] + D[:, i + 1]

    else:
        raise ValueError('Incorrect quad_rule')

    return v_neg, v_pos


def combine_v_neg_pos_to_v(v_neg: np.ndarray,
                           v_pos: np.ndarray,
                           r_m: np.ndarray,
                           N: int,
                           M: int) -> np.ndarray:
    """
    Combine v^- and v^+ into full v with Hermitian symmetry.

    Parameters
    ----------
    v_neg, v_pos : ndarray, shape (N/2+1, M)
        Outputs of compute_v_neg_pos.
    r_m : ndarray, shape (M,)
        Radial grid.
    N, M : int
        Angular and radial counts.

    Returns
    -------
    v : ndarray, shape (N+1, M)
    """
    halfN = N // 2
    v = np.zeros((N + 1, M), dtype=complex)

    # central mode (k = 0)
    v[halfN, 0] = v_neg[halfN, 0] + v_pos[0, 0]
    if M > 1:
        v[halfN, 1:] = np.log(r_m[1:]) * v_neg[halfN, 1:] + v_pos[0, 1:]

    # k = 1..N/2-1 blockwise
    k_idx = np.arange(1, halfN)
    pos_idx = k_idx
    mir_idx = N - k_idx
    pos_from = halfN - k_idx

    v[pos_idx, :] = v_neg[pos_idx, :] + np.conj(v_pos[pos_from, :])
    v[mir_idx, :] = np.conj(v[pos_idx, :])

    return v




