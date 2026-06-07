import cupy as cp

def nonuniform_simps_rule(x: cp.ndarray, f: cp.ndarray) -> float:
    """
    Approximate the definite integral over [x[0], x[2]] of a function sampled
    at three non-uniformly spaced points (x, f), by fitting a quadratic.
    """
    f = f.reshape(-1, 1) if f.ndim == 1 else f  # (3,1)

    x0, x1, x2 = x
    x0_2 = x0 * x0
    x1_2 = x1 * x1
    x2_2 = x2 * x2

    A = cp.array(
        [
            [x0_2, x0, 1.0],
            [x1_2, x1, 1.0],
            [x2_2, x2, 1.0],
        ],
        dtype=float,
    )

    coeff = cp.linalg.solve(A, f)  # (3,1): a, b, c

    a = coeff[0]
    b = coeff[1]
    c = coeff[2]

    result = (
        (a / 3.0) * (x2**3 - x0**3)
        + (b / 2.0) * (x2_2 - x0_2)
        + c * (x2 - x0)
    )

    return result.item()

def compute_C_D_nonuniform(
    r_m: cp.ndarray, f_fourier_coeff: cp.ndarray, quad_rule: int
):
    """
    Compute C and D on a nonuniform radial mesh r_m.
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    C = cp.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = cp.zeros((N // 2 + 1, M - 1), dtype=complex)

    delta = cp.diff(r_m)  # (M-1,)

    if quad_rule == 1:
        # ----- Trapezoidal rule (vectorized) -----
        n = cp.arange(1, N // 2 + 1)[:, None]  # (N//2,1)
        k = -N / 2 + n - 1                     # (N//2,1)

        r_i    = r_m[:-1][None, :]    # (1,M-1)
        r_ip1  = r_m[1:][None, :]     # (1,M-1)
        delta_row = delta[None, :]    # (1,M-1)

        # f slices
        f_pos = f_fourier_coeff[: N // 2, :]      # (N//2,M)
        f_neg = f_fourier_coeff[N // 2 + 1 :, :]  # (N//2,M)

        f_pos_i    = f_pos[:, 1:]
        f_pos_im1 = f_pos[:, :-1]
        f_neg_i    = f_neg[:, 1:]
        f_neg_im1 = f_neg[:, :-1]

        ratio_C = (r_i / r_ip1) ** (-k)
        C[:-1, :] = (delta_row / (4 * k)) * (
            r_i * ratio_C * f_pos_im1 + r_ip1 * f_pos_i
        )

        ratio_D = (r_i / r_ip1) ** n
        D[1:, :] = -(delta_row / (4 * n)) * (
            r_ip1 * ratio_D * f_neg_i + r_i * f_neg_im1
        )

        # Highest frequency n = N//2
        f_max = f_fourier_coeff[N // 2, :]  # (M,)
        C[N // 2, :] = delta * (
            r_m[:-1] * f_max[:-1] + r_m[1:] * f_max[1:]
        ) / 2.0

        # n = 0 mode for D: vectorized
        idx = cp.arange(1, M - 1)
        r_i_vec   = r_m[idx]
        r_ip1_vec = r_m[idx + 1]
        delta_i   = delta[idx]

        term1 = r_ip1_vec * cp.log(r_ip1_vec) * f_max[idx + 1]
        term2 = r_i_vec   * cp.log(r_i_vec)   * f_max[idx]
        D[0, idx] = delta_i / 2.0 * (term1 + term2)

        # i = 0 case
        D[0, 0] = delta[0] / 2.0 * (
            r_m[1] * cp.log(r_m[1]) * f_max[1]
        )

    elif quad_rule == 2:
        # ----- Simpson on nonuniform mesh (vectorized) -----
        halfN = N // 2
        n_arr = cp.arange(1, halfN + 1)
        k_arr = -halfN + n_arr - 1

        # --- Main stencil for interior points i = 1..M-2 ---
        i_vals = cp.arange(1, M - 1)
        r_im1 = r_m[i_vals - 1]
        r_i = r_m[i_vals]
        r_ip1 = r_m[i_vals + 1]

        # Build stack of interpolation matrices A and invert them in a batch
        A_stack = cp.zeros((M - 2, 3, 3), dtype=float)
        A_stack[:, 0, :] = cp.vstack([r_im1**2, r_im1, cp.ones(M - 2)]).T
        A_stack[:, 1, :] = cp.vstack([r_i**2, r_i, cp.ones(M - 2)]).T
        A_stack[:, 2, :] = cp.vstack([r_ip1**2, r_ip1, cp.ones(M - 2)]).T
        A_inv_stack = cp.linalg.inv(A_stack)

        # Integral terms
        dx3 = (r_ip1**3 - r_im1**3)[:, None]
        dx2 = (r_ip1**2 - r_im1**2)[:, None]
        dx1 = (r_ip1 - r_im1)[:, None]

        # --- C and D for modes n = 1..N/2 ---
        if halfN > 0:
            n = n_arr[None, :]
            k = k_arr[None, :]
            f_pos = f_fourier_coeff[:halfN, :]
            f_neg = f_fourier_coeff[halfN + 1:, :]

            # Function values at stencil points for all modes and all i
            f_pos_im1 = f_pos[:, i_vals - 1].T
            f_pos_i = f_pos[:, i_vals].T
            f_pos_ip1 = f_pos[:, i_vals + 1].T
            f_neg_im1 = f_neg[:, i_vals - 1].T
            f_neg_i = f_neg[:, i_vals].T
            f_neg_ip1 = f_neg[:, i_vals + 1].T

            # Reshape for broadcasting
            r_im1_c, r_i_c, r_ip1_c = r_im1[:, None], r_i[:, None], r_ip1[:, None]

            # Build stacks of function values to be integrated
            F_C_stack = cp.empty((M - 2, 3, halfN), dtype=complex)
            F_C_stack[:, 0, :] = (r_im1_c / (2 * k)) * (r_ip1_c / r_im1_c)**k * f_pos_im1
            F_C_stack[:, 1, :] = (r_i_c / (2 * k)) * (r_ip1_c / r_i_c)**k * f_pos_i
            F_C_stack[:, 2, :] = (r_ip1_c / (2 * k)) * f_pos_ip1

            F_D_stack = cp.empty((M - 2, 3, halfN), dtype=complex)
            F_D_stack[:, 0, :] = (-r_im1_c / (2 * n)) * f_neg_im1
            F_D_stack[:, 1, :] = (-r_i_c / (2 * n)) * (r_im1_c / r_i_c)**n * f_neg_i
            F_D_stack[:, 2, :] = (-r_ip1_c / (2 * n)) * (r_im1_c / r_ip1_c)**n * f_neg_ip1

            # Batched solve for quadratic coefficients
            coeff_C = A_inv_stack @ F_C_stack
            coeff_D = A_inv_stack @ F_D_stack

            # Compute integrals from coefficients
            int_C = (coeff_C[:, 0] / 3) * dx3 + (coeff_C[:, 1] / 2) * dx2 + coeff_C[:, 2] * dx1
            int_D = (coeff_D[:, 0] / 3) * dx3 + (coeff_D[:, 1] / 2) * dx2 + coeff_D[:, 2] * dx1

            C[:halfN, 1:] = int_C.T
            D[1:, 1:] = int_D.T

        # --- Endpoint C and D (column 0) using Trapezoidal rule ---
        C[:-1, 0] = (delta[0]**2 / (4.0 * k_arr)) * f_fourier_coeff[:halfN, 1]
        D[1:, 0] = -(delta[M - 2] / (4.0 * n_arr)) * (
            r_m[M - 2] * f_fourier_coeff[halfN + 1:, M - 2] +
            r_m[M - 1] * (r_m[M - 2] / r_m[M - 1])**n_arr * f_fourier_coeff[halfN + 1:, M - 1]
        )

        # --- Highest frequency mode n=N/2 for C ---
        f_max = f_fourier_coeff[halfN, :]
        f_trip_Cmax = cp.vstack([
            r_im1 * f_max[i_vals - 1],
            r_i * f_max[i_vals],
            r_ip1 * f_max[i_vals + 1]
        ]).T
        coeff_max = cp.einsum('ijk,ik->ij', A_inv_stack, f_trip_Cmax)
        int_Cmax = (coeff_max[:, 0] / 3) * dx3.ravel() + (coeff_max[:, 1] / 2) * dx2.ravel() + coeff_max[:, 2] * dx1.ravel()
        C[halfN, 1:] = int_Cmax
        C[N // 2, 0] = (r_m[1] ** 2 / 2.0) * f_fourier_coeff[N // 2, 1]

        # --- Zero mode n=0 for D (with logs) ---
        if M > 3:
            i_log = cp.arange(2, M - 1)
            r_log_im1, r_log_i, r_log_ip1 = r_m[i_log - 1], r_m[i_log], r_m[i_log + 1]
            A_log_stack = cp.zeros((M - 3, 3, 3), dtype=float)
            A_log_stack[:, 0, :] = cp.vstack([r_log_im1**2, r_log_im1, cp.ones(M - 3)]).T
            A_log_stack[:, 1, :] = cp.vstack([r_log_i**2, r_log_i, cp.ones(M - 3)]).T
            A_log_stack[:, 2, :] = cp.vstack([r_log_ip1**2, r_log_ip1, cp.ones(M - 3)]).T
            A_inv_log_stack = cp.linalg.inv(A_log_stack)

            f_trip_D0 = cp.vstack([
                r_log_im1 * cp.log(r_log_im1) * f_max[i_log - 1],
                r_log_i * cp.log(r_log_i) * f_max[i_log],
                r_log_ip1 * cp.log(r_log_ip1) * f_max[i_log + 1]
            ]).T
            coeff_D0 = cp.einsum('ijk,ik->ij', A_inv_log_stack, f_trip_D0)
            dx3_log = (r_log_ip1**3 - r_log_im1**3)
            dx2_log = (r_log_ip1**2 - r_log_im1**2)
            dx1_log = (r_log_ip1 - r_log_im1)
            D[0, 2:] = (coeff_D0[:, 0] / 3) * dx3_log + (coeff_D0[:, 1] / 2) * dx2_log + coeff_D0[:, 2] * dx1_log

        # Edge cases for D[0,:]
        r_trip0 = cp.array([r_m[0], r_m[1], r_m[2]])
        f_trip_D0 = cp.array([
            0.0,
            r_m[1] * cp.log(r_m[1]) * f_fourier_coeff[N // 2, 1],
            r_m[2] * cp.log(r_m[2]) * f_fourier_coeff[N // 2, 2],
        ], dtype=complex)
        if M > 2:
            D[0, 1] = nonuniform_simps_rule(r_trip0, f_trip_D0)

        D[0, 0] = delta[M - 2] / 2.0 * (
            r_m[M - 2] * cp.log(r_m[M - 2]) * f_fourier_coeff[N // 2, M - 2]
            + r_m[M - 1] * cp.log(r_m[M - 1]) * f_fourier_coeff[N // 2, M - 1]
        )

    else:
        raise ValueError("quad_rule must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
