def nonuniform_simps_rule(x: np.ndarray, f: np.ndarray) -> float:
    """
    Approximate the definite integral over [x[0], x[2]] of a function sampled
    at three non-uniformly spaced points (x, f), by fitting a quadratic.
    """
    f = f.reshape(-1, 1) if f.ndim == 1 else f  # (3,1)

    x0, x1, x2 = x
    x0_2 = x0 * x0
    x1_2 = x1 * x1
    x2_2 = x2 * x2

    A = np.array(
        [
            [x0_2, x0, 1.0],
            [x1_2, x1, 1.0],
            [x2_2, x2, 1.0],
        ],
        dtype=float,
    )

    coeff = np.linalg.solve(A, f)  # (3,1): a, b, c

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
    r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int
):
    """
    Compute C and D on a nonuniform radial mesh r_m.
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    delta = np.diff(r_m)  # (M-1,)

    if quad_rule == 1:
        # ----- Trapezoidal rule (vectorized) -----
        n = np.arange(1, N // 2 + 1)[:, None]  # (N//2,1)
        k = -N / 2 + n - 1                     # (N//2,1)

        r_i    = r_m[:-1][None, :]    # (1,M-1)
        r_ip1  = r_m[1:][None, :]     # (1,M-1)
        delta_row = delta[None, :]    # (1,M-1)

        # f slices
        f_pos = f_fourier_coeff[: N // 2, :]      # (N//2,M)
        f_neg = f_fourier_coeff[N // 2 + 1 :, :]  # (N//2,M)

        f_pos_i    = f_pos[:, :-1]
        f_pos_ip1  = f_pos[:, 1:]
        f_neg_i    = f_neg[:, :-1]
        f_neg_ip1  = f_neg[:, 1:]

        ratio_C = (r_i / r_ip1) ** (-k)
        C[:-1, :] = (delta_row / (4 * k)) * (
            r_i * ratio_C * f_pos_i + r_ip1 * f_pos_ip1
        )

        ratio_D = (r_i / r_ip1) ** n
        D[1:, :] = -(delta_row / (4 * n)) * (
            r_ip1 * ratio_D * f_neg_ip1 + r_i * f_neg_i
        )

        # Highest frequency n = N//2
        f_max = f_fourier_coeff[N // 2, :]  # (M,)
        C[N // 2, :] = delta * (
            r_m[:-1] * f_max[:-1] + r_m[1:] * f_max[1:]
        ) / 2.0

        # n = 0 mode for D: vectorized
        # indices i = 1..M-2 (matching original loop range)
        idx = np.arange(1, M - 1)
        r_i_vec   = r_m[idx]
        r_ip1_vec = r_m[idx + 1]
        delta_i   = delta[idx]

        term1 = r_ip1_vec * np.log(r_ip1_vec) * f_max[idx + 1]
        term2 = r_i_vec   * np.log(r_i_vec)   * f_max[idx]
        D[0, idx] = delta_i / 2.0 * (term1 + term2)

        # i = 0 case
        D[0, 0] = delta[0] / 2.0 * (
            r_m[1] * np.log(r_m[1]) * f_max[1]
        )

    elif quad_rule == 2:
        # ----- Simpson on nonuniform mesh (batched in n per i) -----
        n_arr = np.arange(1, N // 2 + 1)       # 1..N/2
        k_arr = -N / 2 + n_arr - 1            # k = -N/2 + n - 1

        idx_pos = np.arange(0, N // 2)         # n-1
        idx_neg = np.arange(N // 2 + 1, N + 1) # n+N//2

        for i in range(1, M - 1):
            r_im1 = r_m[i - 1]
            r_i   = r_m[i]
            r_ip1 = r_m[i + 1]

            x0, x1, x2 = r_im1, r_i, r_ip1
            x0_2, x1_2, x2_2 = x0 * x0, x1 * x1, x2 * x2

            A = np.array(
                [
                    [x0_2, x0, 1.0],
                    [x1_2, x1, 1.0],
                    [x2_2, x2, 1.0],
                ],
                dtype=float,
            )
            A_inv = np.linalg.inv(A)

            f_pos_im1 = f_fourier_coeff[idx_pos, i - 1]
            f_pos_i   = f_fourier_coeff[idx_pos, i]
            f_pos_ip1 = f_fourier_coeff[idx_pos, i + 1]

            f_neg_im1 = f_fourier_coeff[idx_neg, i - 1]
            f_neg_i   = f_fourier_coeff[idx_neg, i]
            f_neg_ip1 = f_fourier_coeff[idx_neg, i + 1]

            k = k_arr.astype(float)
            n = n_arr.astype(float)

            F_C = np.empty((3, N // 2), dtype=complex)
            F_C[0, :] = (
                r_im1 / (2.0 * k)
                * (r_ip1 / r_im1) ** k
                * f_pos_im1
            )
            F_C[1, :] = (
                r_i / (2.0 * k)
                * (r_ip1 / r_i) ** k
                * f_pos_i
            )
            F_C[2, :] = (
                r_ip1 / (2.0 * k)
                * f_pos_ip1
            )

            F_D = np.empty((3, N // 2), dtype=complex)
            F_D[0, :] = (
                -r_im1 / (2.0 * n)
                * f_neg_im1
            )
            F_D[1, :] = (
                -r_i / (2.0 * n)
                * (r_im1 / r_i) ** n
                * f_neg_i
            )
            F_D[2, :] = (
                -r_ip1 / (2.0 * n)
                * (r_im1 / r_ip1) ** n
                * f_neg_ip1
            )

            coeff_C = A_inv @ F_C
            coeff_D = A_inv @ F_D

            a_C, b_C, c_C = coeff_C[0, :], coeff_C[1, :], coeff_C[2, :]
            a_D, b_D, c_D = coeff_D[0, :], coeff_D[1, :], coeff_D[2, :]

            dx3 = x2**3 - x0**3
            dx2 = x2_2 - x0_2
            dx1 = x2 - x0

            int_C = (a_C / 3.0) * dx3 + (b_C / 2.0) * dx2 + c_C * dx1
            int_D = (a_D / 3.0) * dx3 + (b_D / 2.0) * dx2 + c_D * dx1

            C[:-1, i] = int_C
            D[1:,  i] = int_D

            # highest frequency n = N//2 for C
            f_trip_Cmax = np.array([
                r_im1 * f_fourier_coeff[N // 2, i - 1],
                r_i   * f_fourier_coeff[N // 2, i],
                r_ip1 * f_fourier_coeff[N // 2, i + 1],
            ], dtype=complex)
            coeff_max = A_inv @ f_trip_Cmax.reshape(3, 1)
            aM, bM, cM = coeff_max[0, 0], coeff_max[1, 0], coeff_max[2, 0]
            C[N // 2, i] = (
                (aM / 3.0) * dx3 + (bM / 2.0) * dx2 + cM * dx1
            )

            # n = 0 for D (log weights), skip i=1 as in original
            if i != 1:
                f_trip_D0 = np.array([
                    r_im1 * np.log(r_im1) * f_fourier_coeff[N // 2, i - 1],
                    r_i   * np.log(r_i)   * f_fourier_coeff[N // 2, i],
                    r_ip1 * np.log(r_ip1) * f_fourier_coeff[N // 2, i + 1],
                ], dtype=complex)
                coeff_D0 = A_inv @ f_trip_D0.reshape(3, 1)
                a0, b0, c0 = coeff_D0[0, 0], coeff_D0[1, 0], coeff_D0[2, 0]
                D[0, i] = (
                    (a0 / 3.0) * dx3 + (b0 / 2.0) * dx2 + c0 * dx1
                )

            # left-end corrections for C_{1,2}^n, D_{1,2}^n
            if i == 1:
                C[:-1, 0] = (delta[0] ** 2 / (4.0 * k_arr)) * f_fourier_coeff[: N // 2, 1]
                D[1:, 0] = -(delta[M - 2] / (4.0 * n_arr)) * (
                    r_m[M - 2] * f_fourier_coeff[N // 2 + 1 :, M - 2]
                    + r_m[M - 1]
                    * (r_m[M - 2] / r_m[M - 1]) ** n_arr
                    * f_fourier_coeff[N // 2 + 1 :, M - 1]
                )

        # Endpoint corrections for n=N//2 and n=0 (unchanged)
        C[N // 2, 0] = (r_m[1] ** 2 / 2.0) * f_fourier_coeff[N // 2, 1]
        r_trip0 = np.array([r_m[0], r_m[1], r_m[2]])
        f_trip_D0 = np.array([
            0.0,
            r_m[1] * np.log(r_m[1]) * f_fourier_coeff[N // 2, 1],
            r_m[2] * np.log(r_m[2]) * f_fourier_coeff[N // 2, 2],
        ], dtype=complex)
        D[0, 1] = nonuniform_simps_rule(r_trip0, f_trip_D0)
        D[0, 0] = delta[M - 2] / 2.0 * (
            r_m[M - 2] * np.log(r_m[M - 2]) * f_fourier_coeff[N // 2, M - 2]
            + r_m[M - 1] * np.log(r_m[M - 1]) * f_fourier_coeff[N // 2, M - 1]
        )

    else:
        raise ValueError("quad_rule must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
