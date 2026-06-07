
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
        # Simpson variant: 3-point from Borges-Daripa.
        # This is a temporary, slow, loop-based implementation to fix runtime errors.
        # It should be vectorized for performance.
        halfN = N // 2
        n_arr = cp.arange(1, halfN + 1)
        k_arr = -halfN + n_arr - 1

        for i in range(2, M):
            for n_idx, n in enumerate(n_arr):
                k = k_arr[n_idx]
                if k != 0:
                    C[n_idx, i - 1] = (delta**2 / (6 * k)) * (
                        (i - 2) * ((i - 2) / i)**(-k) * f_fourier_coeff[n_idx, i - 2] +
                        4 * (i - 1) * ((i - 1) / i)**(-k) * f_fourier_coeff[n_idx, i - 1] +
                        i * f_fourier_coeff[n_idx, i]
                    )
                D[n_idx + 1, i - 1] = -(delta**2 / (6 * n)) * (
                    (i - 2) * f_fourier_coeff[n_idx + halfN + 1, i - 2] +
                    4 * (i - 1) * ((i - 2) / (i - 1))**n * f_fourier_coeff[n_idx + halfN + 1, i - 1] +
                    i * ((i - 2) / i)**n * f_fourier_coeff[n_idx + halfN + 1, i]
                )

            # Highest frequency mode n=N/2
            C[halfN, i - 1] = (delta**2 / 3.0) * (
                (i - 2) * f_fourier_coeff[halfN, i - 2] +
                4 * (i - 1) * f_fourier_coeff[halfN, i - 1] +
                i * f_fourier_coeff[halfN, i]
            )
            # n=0 mode for D (with logs)
            if i > 2:
                D[0, i - 1] = (delta**2 / 3.0) * (
                    (i - 2) * cp.log((i - 2) * delta) * f_fourier_coeff[halfN, i - 2] +
                    4 * (i - 1) * cp.log((i - 1) * delta) * f_fourier_coeff[halfN, i - 1] +
                    i * cp.log(i * delta) * f_fourier_coeff[halfN, i]
                )

        # Edge cases
        for n_idx, n in enumerate(n_arr):
            k = k_arr[n_idx]
            if k != 0:
                C[n_idx, 0] = (delta**2 / (4 * k)) * f_fourier_coeff[n_idx, 1]
            D[n_idx + 1, 0] = -(delta**2 / (4 * n)) * (
                (M - 1) * ((M - 2) / (M - 1))**n * f_fourier_coeff[n_idx + halfN + 1, M - 1] +
                (M - 2) * f_fourier_coeff[n_idx + halfN + 1, M - 2]
            )

        C[halfN, 0] = (delta**2 / 2.0) * f_fourier_coeff[halfN, 1]
        if M > 2:
            D[0, 1] = (delta**2 / 3.0) * (
                4 * cp.log(delta) * f_fourier_coeff[halfN, 1] +
                2 * cp.log(2 * delta) * f_fourier_coeff[halfN, 2]
            )
        D[0, 0] = (delta**2 / 2.0) * (
            (M - 2) * cp.log((M - 2) * delta) * f_fourier_coeff[halfN, M - 2] +
            (M - 1) * cp.log((M - 1) * delta) * f_fourier_coeff[halfN, M - 1]
        )

    else:
        raise ValueError("Unknown quad_rule; must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
