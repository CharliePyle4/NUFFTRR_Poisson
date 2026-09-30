import numpy as np

def nonuniform_simps_rule(x: np.ndarray, f: np.ndarray) -> float:
    """
    Approximate the definite integral over [x[0], x[2]] of a function sampled
    at three non-uniformly spaced points (x, f), by fitting a quadratic.
    """
    f = f.reshape(-1, 1) if f.ndim == 1 else f

    x0, x1, x2 = x[0], x[1], x[2]
    A = np.array([
        [x0**2, x0, 1.0],
        [x1**2, x1, 1.0],
        [x2**2, x2, 1.0],
    ], dtype=float)

    try:
        coeff = np.linalg.solve(A, f)
    except np.linalg.LinAlgError:
        # Fallback for stability if matrix is nearly singular
        coeff = np.linalg.lstsq(A, f, rcond=None)[0]

    a, b, c = coeff.ravel()

    # Analytically integrate the quadratic a*r^2 + b*r + c from x[0] to x[2].
    result = (
        (a / 3.0) * (x2**3 - x0**3)
        + (b / 2.0) * (x2**2 - x0**2)
        + c * (x2 - x0)
    )

    return result.item()


def compute_C_D_nonuniform(
    r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int
):
    """
    Compute C and D on a nonuniform radial mesh r_m.
    This is a fully vectorized implementation for the CPU.
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1
    halfN = N // 2

    C = np.zeros((halfN + 1, M - 1), dtype=complex)
    D = np.zeros((halfN + 1, M - 1), dtype=complex)

    delta = np.diff(r_m)  # (M-1,)

    if quad_rule == 1:
        # ----- Trapezoidal rule (vectorized) -----
        # Note: k corresponds to negative frequencies, n to positive
        k_slice = np.arange(0, halfN)
        k = -halfN + k_slice
        n = np.arange(1, halfN + 1)

        # Reshape for broadcasting
        k = k[:, None] # (N/2, 1)
        n = n[:, None] # (N/2, 1)

        r_i    = r_m[:-1][None, :]    # (1, M-1)
        r_ip1  = r_m[1:][None, :]     # (1, M-1)
        delta_row = delta[None, :]    # (1, M-1)

        # f slices
        f_pos = f_fourier_coeff[:halfN, :]      # modes k=-N/2..-1
        f_neg = f_fourier_coeff[halfN + 1:, :]  # modes n=1..N/2 (shifted)
        f_max = f_fourier_coeff[halfN, :]       # mode k=0

        f_pos_i    = f_pos[:, :-1]
        f_pos_ip1  = f_pos[:, 1:]
        f_neg_i    = f_neg[:, :-1]
        f_neg_ip1  = f_neg[:, 1:]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_C = (r_i / r_ip1) ** (-k)
            C[:halfN, :] = (delta_row / (4 * k)) * (
                r_i * ratio_C * f_pos_i + r_ip1 * f_pos_ip1
            )

            ratio_D = (r_i / r_ip1) ** n
            D[1:, :] = -(delta_row / (4 * n)) * (
                r_ip1 * ratio_D * f_neg_ip1 + r_i * f_neg_i
            )

        # k=0 mode for C (n=N/2)
        C[halfN, :] = delta * (
            r_m[:-1] * f_max[:-1] + r_m[1:] * f_max[1:]
        ) / 2.0

        # n=0 mode for D (log term)
        with np.errstate(invalid='ignore'): # handle log(0)
            # Interior points
            idx = np.arange(1, M - 1)
            term1 = r_m[idx+1] * np.log(r_m[idx+1]) * f_max[idx+1]
            term2 = r_m[idx] * np.log(r_m[idx]) * f_max[idx]
            D[0, idx] = delta[idx] / 2.0 * (term1 + term2)
            # Left endpoint (i=0)
            D[0, 0] = delta[0] / 2.0 * (r_m[1] * np.log(r_m[1]) * f_max[1])

    elif quad_rule == 2:
        # ----- Simpson on nonuniform mesh (vectorized via analytical weights) -----
        n_arr = np.arange(1, halfN + 1)
        k_arr = -halfN + n_arr - 1

        # --- Main stencil for interior points i = 1..M-2 ---
        i_vals = np.arange(1, M - 1)
        r_im1 = r_m[i_vals - 1]
        r_i = r_m[i_vals]
        r_ip1 = r_m[i_vals + 1]

        # Analytical nonuniform Simpson quadrature weights for [r_im1, r_ip1]
        h0 = r_i - r_im1
        h1 = r_ip1 - r_i
        H = h0 + h1
        w0 = (H * (2.0 * h0 - h1)) / (6.0 * h0)
        w1 = (H**3) / (6.0 * h0 * h1)
        w2 = (H * (2.0 * h1 - h0)) / (6.0 * h1)

        # --- C and D for modes n = 1..N/2 ---
        if halfN > 0:
            n = n_arr[None, :]  # (1, N/2)
            k = k_arr[None, :]  # (1, N/2)
            f_pos = f_fourier_coeff[:halfN, :]
            f_neg = f_fourier_coeff[halfN + 1:, :]

            # Function values at stencil points for all modes and all i
            f_pos_im1 = f_pos[:, i_vals - 1].T
            f_pos_i = f_pos[:, i_vals].T
            f_pos_ip1 = f_pos[:, i_vals + 1].T
            f_neg_im1 = f_neg[:, i_vals - 1].T
            f_neg_i = f_neg[:, i_vals].T
            f_neg_ip1 = f_neg[:, i_vals + 1].T

            # Reshape for broadcasting (M-2, 1)
            r_im1_c, r_i_c, r_ip1_c = r_im1[:, None], r_i[:, None], r_ip1[:, None]

            with np.errstate(divide='ignore', invalid='ignore'):
                F_C_0 = (r_im1_c / (2 * k)) * (r_im1_c / r_ip1_c)**(-k) * f_pos_im1
                F_C_1 = (r_i_c / (2 * k)) * (r_i_c / r_ip1_c)**(-k) * f_pos_i
                F_C_2 = (r_ip1_c / (2 * k)) * f_pos_ip1

                F_D_0 = (-r_im1_c / (2 * n)) * f_neg_im1
                F_D_1 = (-r_i_c / (2 * n)) * (r_im1_c / r_i_c)**n * f_neg_i
                F_D_2 = (-r_ip1_c / (2 * n)) * (r_im1_c / r_ip1_c)**n * f_neg_ip1

            int_C = w0[:, None] * F_C_0 + w1[:, None] * F_C_1 + w2[:, None] * F_C_2
            int_D = w0[:, None] * F_D_0 + w1[:, None] * F_D_1 + w2[:, None] * F_D_2

            C[:halfN, 1:] = int_C.T
            D[1:, 1:] = int_D.T

        # --- Endpoint C and D (column 0) using Trapezoidal rule ---
        if halfN > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                C[:halfN, 0] = (delta[0]**2 / (4.0 * k_arr)) * f_fourier_coeff[:halfN, 1]
                D[1:, 0] = -(delta[M - 2] / (4.0 * n_arr)) * (
                    r_m[M - 2] * f_fourier_coeff[halfN + 1:, M - 2] +
                    r_m[M - 1] * (r_m[M - 2] / r_m[M - 1])**n_arr * f_fourier_coeff[halfN + 1:, M - 1]
                )

        # --- Highest frequency mode n=N/2 for C ---
        f_max = f_fourier_coeff[halfN, :]
        int_Cmax = (
            w0 * (r_im1 * f_max[i_vals - 1])
            + w1 * (r_i * f_max[i_vals])
            + w2 * (r_ip1 * f_max[i_vals + 1])
        )
        C[halfN, 1:] = int_Cmax
        C[halfN, 0] = (r_m[1]**2 / 2.0) * f_fourier_coeff[halfN, 1]

        # --- Zero mode n=0 for D (with logs) ---
        with np.errstate(invalid='ignore'):
            if M > 3:
                i_log = np.arange(2, M - 1)
                w0_log, w1_log, w2_log = w0[1:], w1[1:], w2[1:]
                r_log_im1, r_log_i, r_log_ip1 = r_im1[1:], r_i[1:], r_ip1[1:]

                D[0, 2:] = (
                    w0_log * (r_log_im1 * np.log(r_log_im1) * f_max[i_log - 1])
                    + w1_log * (r_log_i * np.log(r_log_i) * f_max[i_log])
                    + w2_log * (r_log_ip1 * np.log(r_log_ip1) * f_max[i_log + 1])
                )

            # Edge cases for D[0,:]
            if M > 2:
                h0_e = r_m[1] - r_m[0]
                h1_e = r_m[2] - r_m[1]
                H_e = h0_e + h1_e
                w1_e = (H_e**3) / (6.0 * h0_e * h1_e)
                w2_e = (H_e * (2.0 * h1_e - h0_e)) / (6.0 * h1_e)
                D[0, 1] = (
                    w1_e * (r_m[1] * np.log(r_m[1]) * f_fourier_coeff[halfN, 1])
                    + w2_e * (r_m[2] * np.log(r_m[2]) * f_fourier_coeff[halfN, 2])
                )

            # D^(M-1,M) for n=0 mode (trapezoidal rule on last interval)
            if M > 1:
                D[0, 0] = delta[M - 2] / 2.0 * (
                    r_m[M - 2] * np.log(r_m[M - 2]) * f_fourier_coeff[halfN, M - 2] +
                    r_m[M - 1] * np.log(r_m[M - 1]) * f_fourier_coeff[halfN, M - 1]
                )

    else:
        raise ValueError("quad_rule must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
