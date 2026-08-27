
import cupy as cp

# ---------------------------------------------------------
# Fused CUDA Elementwise Kernels for Radial Quadratures
# (All index variables are strictly typed as float64 to ensure
#  IEEE-754 64-bit floating point division in CUDA C++)
# ---------------------------------------------------------
@cp.fuse()
def _fuse_trap_C(delta, k, i_prev, i, f_pos_im1, f_pos_i):
    return (delta**2 / (4.0 * k)) * (
        i_prev * ((i_prev / i) ** (-k)) * f_pos_im1 + i * f_pos_i
    )

@cp.fuse()
def _fuse_trap_D(delta, n, i_prev, i, f_neg_i, f_neg_im1):
    return -(delta**2 / (4.0 * n)) * (
        i * ((i_prev / i) ** n) * f_neg_i + i_prev * f_neg_im1
    )

@cp.fuse()
def _fuse_trap_C_max(delta, i_prev, f_max_prev, i, f_max_curr):
    return (delta**2 / 2.0) * (i_prev * f_max_prev + i * f_max_curr)

@cp.fuse()
def _fuse_trap_D0(delta, idx, f1, f2):
    term1 = (idx + 1.0) * cp.log((idx + 1.0) * delta) * f1
    term2 = idx * cp.log(idx * delta) * f2
    return (delta**2 / 2.0) * (term1 + term2)

@cp.fuse()
def _fuse_simpson_C(delta, k_col, i_m2, i_m1, i, f_m2, f_m1, f_curr):
    term1 = i_m2 * ((i_m2 / i) ** (-k_col)) * f_m2
    term2 = 4.0 * i_m1 * ((i_m1 / i) ** (-k_col)) * f_m1
    term3 = i * f_curr
    return (delta**2 / (6.0 * k_col)) * (term1 + term2 + term3)

@cp.fuse()
def _fuse_simpson_D(delta, n_col, i_m2, i_m1, i, f_m2, f_m1, f_curr):
    term1 = i_m2 * f_m2
    term2 = 4.0 * i_m1 * ((i_m2 / i_m1) ** n_col) * f_m1
    term3 = i * ((i_m2 / i) ** n_col) * f_curr
    return -(delta**2 / (6.0 * n_col)) * (term1 + term2 + term3)

@cp.fuse()
def _fuse_simpson_C_max(delta, i_m2, f_m2, i_m1, f_m1, i, f_curr):
    return (delta**2 / 3.0) * (i_m2 * f_m2 + 4.0 * i_m1 * f_m1 + i * f_curr)

@cp.fuse()
def _fuse_simpson_D0_log(delta, r_m2, r_m1, r_curr, i_log, f_m2, f_m1, f_curr):
    term1 = (i_log - 2.0) * cp.log(r_m2) * f_m2
    term2 = 4.0 * (i_log - 1.0) * cp.log(r_m1) * f_m1
    term3 = i_log * cp.log(r_curr) * f_curr
    return (delta**2 / 3.0) * (term1 + term2 + term3)


def compute_C_D_uniform(
    r_m: cp.ndarray, f_fourier_coeff: cp.ndarray, quad_rule: int
):
    """
    Compute C and D on a uniform radial mesh r_m (spacing delta).

    quad_rule = 1: trapezoidal (vectorized),
    quad_rule = 2: 3‑point Simpson variant from Borges–Daripa (Sec. 3).
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    C = cp.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = cp.zeros((N // 2 + 1, M - 1), dtype=complex)

    delta = float(r_m[1] - r_m[0])

    if quad_rule == 1:
        # Trapezoidal rule: integer indices for array indexing, float64 for fused math
        i_idx = cp.arange(1, M)          # 1..M-1 (int64)
        i_prev_idx = i_idx - 1

        i_f = i_idx.astype(cp.float64)   # float64 guarantees IEEE-754 float division
        i_prev_f = i_prev_idx.astype(cp.float64)

        n_f = cp.arange(1, N // 2 + 1, dtype=cp.float64)[:, None]
        k_f = -N / 2.0 + n_f - 1.0

        f_pos = f_fourier_coeff[: N // 2, :]
        f_neg = f_fourier_coeff[N // 2 + 1 :, :]

        f_pos_i = f_pos[:, i_idx]
        f_pos_im1 = f_pos[:, i_prev_idx]
        f_neg_i = f_neg[:, i_idx]
        f_neg_im1 = f_neg[:, i_prev_idx]

        C[:-1, :] = _fuse_trap_C(delta, k_f, i_prev_f, i_f, f_pos_im1, f_pos_i)
        D[1:, :] = _fuse_trap_D(delta, n_f, i_prev_f, i_f, f_neg_i, f_neg_im1)

        f_max = f_fourier_coeff[N // 2, :]
        C[N // 2, :] = _fuse_trap_C_max(delta, i_prev_f, f_max[i_prev_idx], i_f, f_max[i_idx])

        # Vectorized calculation for D[0, :] for n=0 mode.
        idx = cp.arange(1, M - 1)
        idx_f = idx.astype(cp.float64)
        D[0, idx] = _fuse_trap_D0(delta, idx_f, f_max[idx + 1], f_max[idx])

        # Handle i=1 case (idx=0) separately.
        D[0, 0] = (delta**2 / 2.0) * (cp.log(delta) * f_max[1])

    elif quad_rule == 2:
        # Simpson variant: 3-point stencil, fused kernel execution
        halfN = N // 2
        f_max = f_fourier_coeff[halfN, :]

        # --- Main stencil for indices i = 2..M-1 ---
        i_idx = cp.arange(2, M)
        i_m1_idx = i_idx - 1
        i_m2_idx = i_idx - 2

        i_f = i_idx.astype(cp.float64)
        i_m1_f = i_m1_idx.astype(cp.float64)
        i_m2_f = i_m2_idx.astype(cp.float64)

        # --- Modes k != 0 and n != 0 ---
        if halfN > 0:
            # Negative frequencies (for C)
            k_vec_f = (cp.arange(0, halfN) - halfN).astype(cp.float64)
            k_col_f = k_vec_f[:, None]
            f_pos = f_fourier_coeff[0:halfN, :]

            # Positive frequencies (for D)
            n_vec_f = cp.arange(1, halfN + 1, dtype=cp.float64)
            n_col_f = n_vec_f[:, None]
            f_neg = f_fourier_coeff[halfN+1 : N+1, :]

            # C calculation for i=2..M-1
            C[0:halfN, i_m1_idx] = _fuse_simpson_C(
                delta, k_col_f, i_m2_f, i_m1_f, i_f,
                f_pos[:, i_m2_idx], f_pos[:, i_m1_idx], f_pos[:, i_idx]
            )

            # D calculation for i=2..M-1
            D[1:halfN+1, i_m1_idx] = _fuse_simpson_D(
                delta, n_col_f, i_m2_f, i_m1_f, i_f,
                f_neg[:, i_m2_idx], f_neg[:, i_m1_idx], f_neg[:, i_idx]
            )

            # Edge cases for index 0 (from original i=2 case)
            C[0:halfN, 0] = (delta**2 / (4.0 * k_vec_f)) * f_pos[:, 1]
            term1_D0 = (M - 1.0) * (((M - 2.0) / (M - 1.0)) ** n_vec_f) * f_neg[:, M - 1]
            term2_D0 = (M - 2.0) * f_neg[:, M - 2]
            D[1:halfN+1, 0] = -(delta**2 / (4.0 * n_vec_f)) * (term1_D0 + term2_D0)

        # --- Highest frequency mode k=0 (n=N/2) for C ---
        C[halfN, i_m1_idx] = _fuse_simpson_C_max(delta, i_m2_f, f_max[i_m2_idx], i_m1_f, f_max[i_m1_idx], i_f, f_max[i_idx])
        C[halfN, 0] = (delta**2 / 2.0) * f_max[1]

        # --- Zero mode k=0 (n=N/2) for D (with logs) ---
        if M > 2:
            i_log = cp.arange(3, M)
            i_log_f = i_log.astype(cp.float64)
            D[0, i_log - 1] = _fuse_simpson_D0_log(
                delta, r_m[i_log - 2], r_m[i_log - 1], r_m[i_log],
                i_log_f, f_max[i_log - 2], f_max[i_log - 1], f_max[i_log]
            )

        # Edge cases for D[0,:]
        if M > 2:
            D[0, 1] = (delta**2 / 3.0) * (
                4.0 * cp.log(delta) * f_max[1] +
                2.0 * cp.log(2.0 * delta) * f_max[2]
            )
        D[0, 0] = (delta**2 / 2.0) * (
            (M - 2.0) * cp.log(r_m[M - 2]) * f_max[M - 2] +
            (M - 1.0) * cp.log(r_m[M - 1]) * f_max[M - 1]
        )

    else:
        raise ValueError("Unknown quad_rule; must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
