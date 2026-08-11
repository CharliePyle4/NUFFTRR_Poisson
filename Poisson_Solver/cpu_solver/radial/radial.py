import numpy as np
import warnings



from .uniform import compute_C_D_uniform
from .nonuniform import compute_C_D_nonuniform

def compute_radial_integrals(r_m: np.ndarray,
                             f_fourier_coeff: np.ndarray,
                             quad_rule: int,
                             rad_unif: int):
    if rad_unif == 1:
        C, D = compute_C_D_uniform(r_m, f_fourier_coeff, quad_rule)
    elif rad_unif == 0:
        C, D = compute_C_D_nonuniform(r_m, f_fourier_coeff, quad_rule)
    else:
        raise ValueError('Incorrect index for "rad_unif"')
    return C, D


def _vectorized_1step_recurrence(a: np.ndarray, Y0: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Helper to solve 1-step linear recurrence y_k = a_k y_{k-1} + C_k in vectorized 2D NumPy."""
    if a.shape[1] == 0:
        return Y0
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        P = np.cumprod(a, axis=1)
        T = np.where(P != 0, C / P, 0.0)
        S = np.cumsum(T, axis=1)
        res = np.nan_to_num(P * (Y0 + S))
        
    return np.hstack([Y0, res])


def _compute_v_neg_pos_numpy(C: np.ndarray,
                      D: np.ndarray,
                      r_m: np.ndarray,
                      N: int,
                      M: int,
                      quad_rule: int):
    halfN = N // 2
    modes = np.arange(halfN + 1)

    v_neg = np.zeros((halfN + 1, M), dtype=complex)
    v_pos = np.zeros((halfN + 1, M), dtype=complex)

    if quad_rule == 1:
        # Trapezoidal: 1‑step recurrences (mode-vectorized)
        exp_neg = (modes - halfN)[:, None]
        exp_pos = modes[:, None]

        if M > 1:
            v_neg[:, 1] = C[:, 0]
        if M > 2:
            r_ratio_neg = (r_m[2:M] / r_m[1:M-1])[None, :] ** exp_neg
            v_neg[:, 1:] = _vectorized_1step_recurrence(r_ratio_neg, C[:, 0:1], C[:, 1:M-1])

        if M > 1:
            v_pos[:, M - 2] = D[:, M - 2]
        if M > 2:
            r_ratio_pos = (r_m[0:M-1] / r_m[1:M])[None, :] ** exp_pos
            a_rev = r_ratio_pos[:, :M-2][:, ::-1]
            C_rev = D[:, :M-2][:, ::-1]
            Y0 = D[:, M-2:M-1]
            res_rev = _vectorized_1step_recurrence(a_rev, Y0, C_rev)
            v_pos[:, :M-1] = res_rev[:, ::-1]

    elif quad_rule == 2:
        # Simpson: 2‑step recurrences (mode-vectorized)
        exp_neg = (modes - halfN)[:, None]
        exp_pos = modes[:, None]

        if M > 1:
            v_neg[:, 1] = C[:, 0]
        if M > 2:
            r_ratio_neg = (r_m[2:M] / r_m[0:M-2])[None, :] ** exp_neg
            
            v_neg[:, 2] = r_ratio_neg[:, 0] * v_neg[:, 0] + C[:, 1]
            
            if M > 4:
                a_even = r_ratio_neg[:, 2::2]
                C_even = C[:, 3:M-1:2]
                res_even = _vectorized_1step_recurrence(a_even, v_neg[:, 2:3], C_even)
                v_neg[:, 2:M:2] = res_even

            if M > 3:
                a_odd = r_ratio_neg[:, 1::2]
                C_odd = C[:, 2:M-1:2]
                res_odd = _vectorized_1step_recurrence(a_odd, v_neg[:, 1:2], C_odd)
                v_neg[:, 1:M:2] = res_odd

        if M > 1:
            v_pos[:, M - 2] = D[:, 0]
        if M > 2:
            r_ratio_pos = (r_m[0:M-2] / r_m[2:M])[None, :] ** exp_pos
            
            a_seq1 = r_ratio_pos[:, M-3:0:-2]
            C_seq1 = D[:, M-2:1:-2]
            Y0_1 = v_pos[:, M-1:M]
            res_seq1 = _vectorized_1step_recurrence(a_seq1, Y0_1, C_seq1)
            v_pos[:, M-1:0:-2] = res_seq1

            a_seq2 = r_ratio_pos[:, M-4:0:-2]
            C_seq2 = D[:, M-3:1:-2]
            Y0_2 = v_pos[:, M-2:M-1]
            res_seq2 = _vectorized_1step_recurrence(a_seq2, Y0_2, C_seq2)
            v_pos[:, M-2:0:-2] = res_seq2

            v_pos[:, 0] = r_ratio_pos[:, 0] * v_pos[:, 2] + D[:, 1]

    else:
        raise ValueError('Incorrect quad_rule')

    return v_neg, v_pos

    return v_neg, v_pos





def compute_v_neg_pos(C: np.ndarray,
                      D: np.ndarray,
                      r_m: np.ndarray,
                      N: int,
                      M: int,
                      quad_rule: int):
    """
    Compute v^- and v^+ via radial recurrences.

    Parameters are passed to the appropriate backend.

    Returns
    -------
    v_neg, v_pos : ndarray, shape (N/2+1, M)
    """
    # Always use the vectorized pure-NumPy backend for zero-loop execution
    return _compute_v_neg_pos_numpy(C, D, r_m, N, M, quad_rule)


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

    # Central mode (k = 0, index halfN)
    # v_0(r) = log(r) * v_0^-(r) + v_0^+(r)
    v[halfN, 0] = v_neg[halfN, 0] + v_pos[0, 0]
    if M > 1:
        # Handle r_m[0] = 0 case for log
        with np.errstate(divide='ignore'):
            log_r = np.log(r_m[1:])
        v[halfN, 1:] = log_r * v_neg[halfN, 1:] + v_pos[0, 1:]

    # The combination formula is v_k = v_k^- + conj(v_{-k}^+)
    # For k < 0, we compute directly.
    # For k > 0, we use Hermitian symmetry v_k = conj(v_{-k}).

    if halfN > 0:
        # All negative modes k = -N/2, ..., -1 (indices 0..halfN-1)
        neg_indices = np.arange(0, halfN)
        # For k = n-halfN, the dual positive k is -k = halfN-n
        pos_dual_indices = halfN - neg_indices
        v[neg_indices, :] = v_neg[neg_indices, :] + np.conj(v_pos[pos_dual_indices, :])

        # All positive modes k = 1, ..., N/2 (indices halfN+1..N)
        pos_indices = np.arange(halfN + 1, N + 1)
        # For k = n-halfN, the dual negative k is -k.
        # The array index for -k is (-k)+halfN = -(n-halfN)+halfN = N-n.
        neg_dual_indices = N - pos_indices
        v[pos_indices, :] = np.conj(v[neg_dual_indices, :])

    return v
