
import numpy as np

def compute_C_D_uniform(
    r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int
):
    """
    Compute C and D on a uniform radial mesh r_m (spacing delta).
    This is a fully vectorized implementation.

    quad_rule = 1: trapezoidal (vectorized),
    quad_rule = 2: 3‑point Simpson variant from Borges–Daripa (Sec. 3).
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1
    halfN = N // 2

    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    if M <= 1:
        return C, D

    delta = r_m[1] - r_m[0]

    if quad_rule == 1:
        # ----- Trapezoidal rule (vectorized) -----
        i = np.arange(1, M)          # 1..M-1
        i_prev = i - 1

        # Modes n=1..N/2
        if halfN > 0:
            n = np.arange(1, halfN + 1)[:, None]
            k = -halfN + n - 1

            f_pos = f_fourier_coeff[:halfN, :]
            f_neg = f_fourier_coeff[halfN + 1:, :]

            f_pos_im1 = f_pos[:, i_prev]
            f_pos_i = f_pos[:, i]
            f_neg_im1 = f_neg[:, i_prev]
            f_neg_i = f_neg[:, i]

            with np.errstate(divide='ignore', invalid='ignore'):
                C[:halfN, :] = (delta**2 / (4 * k)) * (
                    i_prev * ((i_prev / i) ** (-k)) * f_pos_im1 + i * f_pos_i
                )
                D[1:, :] = -(delta**2 / (4 * n)) * (
                    i * ((i_prev / i) ** n) * f_neg_i + i_prev * f_neg_im1
                )

        # Highest frequency n=N/2 for C
        f_max = f_fourier_coeff[halfN, :]
        C[halfN, :] = (delta**2 / 2.0) * (
            i_prev * f_max[i_prev] + i * f_max[i]
        )

        # Zero mode n=0 for D (log term)
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = i * np.log(i * delta) * f_max[i]
            term2 = i_prev * np.log(i_prev * delta) * f_max[i_prev]
        term2[0] = 0.0  # Handle i_prev=0, where 0*log(0) -> 0
        D[0, :] = (delta**2 / 2.0) * (term1 + term2)

    elif quad_rule == 2:
        # ----- Simpson variant (vectorized) -----
        if M < 3:
             # Simpson's rule requires at least 3 points for the main stencil.
             # Fallback to trapezoidal for M=2.
             return compute_C_D_uniform(r_m, f_fourier_coeff, quad_rule=1)

        f_max = f_fourier_coeff[halfN, :]
        i = np.arange(2, M) # Stencil center indices: 2..M-1
        i_out = i - 1       # Output column indices: 1..M-2

        # --- Modes n=1..N/2 ---
        if halfN > 0:
            n = np.arange(1, halfN + 1)[:, None] # (halfN, 1)
            k = -halfN + n - 1                  # (halfN, 1)
            i_b = i[None, :]                    # (1, M-2)

            f_pos = f_fourier_coeff[:halfN, :]
            f_neg = f_fourier_coeff[halfN + 1:, :]

            # C matrix (main part)
            with np.errstate(divide='ignore', invalid='ignore'):
                term1_C = (i_b - 2) * ((i_b - 2) / i_b) ** (-k) * f_pos[:, i - 2]
                term2_C = 4 * (i_b - 1) * ((i_b - 1) / i_b) ** (-k) * f_pos[:, i - 1]
                term3_C = i_b * f_pos[:, i]
                C[:halfN, i_out] = (delta**2 / (6 * k)) * (term1_C + term2_C + term3_C)

            # D matrix (main part)
            term1_D = (i_b - 2) * f_neg[:, i - 2]
            term2_D = 4 * (i_b - 1) * ((i_b - 2) / (i_b - 1)) ** n * f_neg[:, i - 1]
            term3_D = i_b * ((i_b - 2) / i_b) ** n * f_neg[:, i]
            D[1:, i_out] = -(delta**2 / (6 * n)) * (term1_D + term2_D + term3_D)

            # Endpoints (Trapezoidal)
            n_ep = np.arange(1, halfN + 1)
            k_ep = -halfN + n_ep - 1
            with np.errstate(divide='ignore', invalid='ignore'):
                C[:halfN, 0] = (delta**2 / (4 * k_ep)) * f_fourier_coeff[:halfN, 1]
                D[1:, 0] = -(delta**2 / (4 * n_ep)) * (
                    (M - 1) * ((M - 2) / (M - 1)) ** n_ep * f_fourier_coeff[halfN + 1:, M - 1]
                    + (M - 2) * f_fourier_coeff[halfN + 1:, M - 2]
                )

        # --- Highest frequency mode n=N/2 for C ---
        C[halfN, i_out] = (delta**2 / 3.0) * (
            (i - 2) * f_max[i - 2] + 4 * (i - 1) * f_max[i - 1] + i * f_max[i]
        )
        C[halfN, 0] = (delta**2 / 2.0) * f_max[1] # Trapezoidal endpoint

        # --- Zero mode n=0 for D (log term) ---
        with np.errstate(divide='ignore', invalid='ignore'):
            # Main part (i=3..M-1)
            if M > 3:
                i_log = np.arange(3, M)
                i_log_out = i_log - 1
                term1 = (i_log - 2) * np.log((i_log - 2) * delta) * f_max[i_log - 2]
                term2 = 4 * (i_log - 1) * np.log((i_log - 1) * delta) * f_max[i_log - 1]
                term3 = i_log * np.log(i_log * delta) * f_max[i_log]
                D[0, i_log_out] = (delta**2 / 3.0) * (term1 + term2 + term3)

            # Special case for D[0, 1] (i=2), where 0*log(0) -> 0
            D[0, 1] = (delta**2 / 3.0) * (
                4 * np.log(delta) * f_max[1] + 2 * np.log(2 * delta) * f_max[2]
            )
            # Special case for D[0, 0] (Trapezoidal on last interval)
            D[0, 0] = (delta**2 / 2.0) * (
                (M - 2) * np.log((M - 2) * delta) * f_max[M - 2]
                + (M - 1) * np.log((M - 1) * delta) * f_max[M - 1]
            )

    else:
        raise ValueError("Unknown quad_rule; must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
