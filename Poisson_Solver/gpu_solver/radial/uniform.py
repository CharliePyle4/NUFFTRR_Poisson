
import cupy as cp

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

    delta = r_m[1] - r_m[0]

    if quad_rule == 1:
        # Trapezoidal rule (vectorized)
        i = cp.arange(1, M)          # 1..M-1
        i_prev = i - 1

        n = cp.arange(1, N // 2 + 1)[:, None]
        k = -N / 2 + n - 1

        f_pos = f_fourier_coeff[: N // 2, :]
        f_neg = f_fourier_coeff[N // 2 + 1 :, :]

        f_pos_im1 = f_pos[:, i_prev]
        f_pos_i = f_pos[:, i]
        f_neg_im1 = f_neg[:, i_prev]
        f_neg_i = f_neg[:, i]

        C[:-1, :] = (delta**2 / (4 * k)) * (
            i_prev * ((i_prev / i) ** (-k)) * f_pos_im1 + i * f_pos_i
        )
        D[1:, :] = -(delta**2 / (4 * n)) * (
            i * ((i_prev / i) ** n) * f_neg_i + i_prev * f_neg_im1
        )

        f_max = f_fourier_coeff[N // 2, :]
        C[N // 2, :] = (delta**2 / 2.0) * (
            i_prev * f_max[i_prev] + i * f_max[i]
        )

        # Vectorized calculation for D[0, :] for n=0 mode.
        # This handles indices idx = 1..M-2, which corresponds to i=2..M-1.
        idx = cp.arange(1, M - 1)
        term1 = (idx + 1) * cp.log((idx + 1) * delta) * f_max[idx + 1]
        term2 = idx * cp.log(idx * delta) * f_max[idx]
        D[0, idx] = (delta**2 / 2.0) * (term1 + term2)

        # Handle i=1 case (idx=0) separately.
        D[0, 0] = (delta**2 / 2.0) * (cp.log(delta) * f_max[1])

    elif quad_rule == 2:
        # Simpson variant: 3-point stencil, fully vectorized.
        halfN = N // 2
        f_max = f_fourier_coeff[halfN, :]

        # --- Main stencil for indices i = 2..M-1 ---
        i = cp.arange(2, M)
        i_m1 = i - 1
        i_m2 = i - 2

        # --- Modes k = -N/2 .. -1 and k = 1 .. N/2 ---
        if halfN > 0:
            # Negative frequencies (for C)
            k_vec = cp.arange(0, halfN) - halfN  # k = -N/2, ..., -1
            k_col = k_vec[:, None] # shape (halfN, 1)
            f_pos = f_fourier_coeff[0:halfN, :] # f_n for n=0..halfN-1

            # Positive frequencies (for D)
            n_vec = cp.arange(1, halfN + 1) # n = 1, ..., N/2
            n_col = n_vec[:, None] # shape (halfN, 1)
            f_neg = f_fourier_coeff[halfN+1 : N+1, :] # f_n for n=halfN+1..N

            # C calculation for i=2..M-1
            term1_C = (i_m2) * ((i_m2 / i) ** (-k_col)) * f_pos[:, i_m2]
            term2_C = 4 * (i_m1) * ((i_m1 / i) ** (-k_col)) * f_pos[:, i_m1]
            term3_C = i * f_pos[:, i]
            C[0:halfN, i_m1] = (delta**2 / (6 * k_col)) * (term1_C + term2_C + term3_C)

            # D calculation for i=2..M-1
            term1_D = (i_m2) * f_neg[:, i_m2]
            term2_D = 4 * (i_m1) * ((i_m2 / i_m1) ** n_col) * f_neg[:, i_m1]
            term3_D = i * ((i_m2 / i) ** n_col) * f_neg[:, i]
            D[1:halfN+1, i_m1] = -(delta**2 / (6 * n_col)) * (term1_D + term2_D + term3_D)

            # Edge cases for index 0 (from original i=2 case)
            C[0:halfN, 0] = (delta**2 / (4 * k_vec)) * f_pos[:, 1]
            term1_D0 = (M - 1) * (((M - 2) / (M - 1)) ** n_col) * f_neg[:, M - 1]
            term2_D0 = (M - 2) * f_neg[:, M - 2]
            D[1:halfN+1, 0] = -(delta**2 / (4 * n_col)) * (term1_D0 + term2_D0)

        # --- Highest frequency mode k=0 (n=N/2) for C ---
        C[halfN, i_m1] = (delta**2 / 3.0) * (i_m2 * f_max[i_m2] + 4 * i_m1 * f_max[i_m1] + i * f_max[i])
        C[halfN, 0] = (delta**2 / 2.0) * f_max[1]

        # --- Zero mode k=0 (n=N/2) for D (with logs) ---
        if M > 2:
            i_log = cp.arange(3, M)
            term1_D0_log = (i_log - 2) * cp.log(r_m[i_log - 2]) * f_max[i_log - 2]
            term2_D0_log = 4 * (i_log - 1) * cp.log(r_m[i_log - 1]) * f_max[i_log - 1]
            term3_D0_log = i_log * cp.log(r_m[i_log]) * f_max[i_log]
            D[0, i_log - 1] = (delta**2 / 3.0) * (term1_D0_log + term2_D0_log + term3_D0_log)

        # Edge cases for D[0,:]
        if M > 2:
            D[0, 1] = (delta**2 / 3.0) * (
                4 * cp.log(delta) * f_max[1] +
                2 * cp.log(2 * delta) * f_max[2]
            )
        D[0, 0] = (delta**2 / 2.0) * (
            (M - 2) * cp.log(r_m[M - 2]) * f_max[M - 2] +
            (M - 1) * cp.log(r_m[M - 1]) * f_max[M - 1]
        )

    else:
        raise ValueError("Unknown quad_rule; must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
