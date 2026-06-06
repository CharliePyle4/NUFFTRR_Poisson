
import numpy as np

def compute_C_D_uniform(
    r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int
):
    """
    Compute C and D on a uniform radial mesh r_m (spacing delta).

    quad_rule = 1: trapezoidal (vectorized),
    quad_rule = 2: 3‑point Simpson variant from Borges–Daripa (Sec. 3).
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    delta = r_m[1] - r_m[0]

    if quad_rule == 1:
        # your existing trapezoidal implementation (unchanged)
        i = np.arange(1, M)          # 1..M-1
        i_prev = i - 1

        n = np.arange(1, N // 2 + 1)[:, None]
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

        for idx, ii in enumerate(i):
            if ii != 1:
                D[0, idx] = (delta**2 / 2.0) * (
                    ii * np.log(ii * delta) * f_max[ii]
                    + (ii - 1) * np.log((ii - 1) * delta) * f_max[ii - 1]
                )
        D[0, 0] = (delta**2 / 2.0) * (np.log(delta) * f_max[1])

    elif quad_rule == 2:
        # Simpson variant: 3‑point C_{i-1,i+1}, D_{i-1,i+1}
        for i in range(2, M):  # i=2..M-1
            for n in range(1, N // 2 + 1):
                k = -N / 2 + n - 1

                C[n - 1, i - 1] = (delta**2 / (6 * k)) * (
                    (i - 2)
                    * ((i - 2) / i) ** (-k)
                    * f_fourier_coeff[n - 1, i - 2]
                    + 4 * (i - 1)
                    * ((i - 1) / i) ** (-k)
                    * f_fourier_coeff[n - 1, i - 1]
                    + i * f_fourier_coeff[n - 1, i]
                )

                D[n, i - 1] = -(delta**2 / (6 * n)) * (
                    (i - 2) * f_fourier_coeff[n + N // 2, i - 2]
                    + 4 * (i - 1)
                    * ((i - 2) / (i - 1)) ** n
                    * f_fourier_coeff[n + N // 2, i - 1]
                    + i * ((i - 2) / i) ** n
                    * f_fourier_coeff[n + N // 2, i]
                )

                if i == 2:
                    # left endpoint C_{1,2}^n, D_{1,2}^n
                    C[n - 1, 0] = (delta**2 / (4 * k)) * f_fourier_coeff[n - 1, 1]
                    D[n, 0] = -(delta**2 / (4 * n)) * (
                        (M - 1)
                        * ((M - 2) / (M - 1)) ** n
                        * f_fourier_coeff[n + N // 2, M - 1]
                        + (M - 2) * f_fourier_coeff[n + N // 2, M - 2]
                    )

            # highest frequency n=N//2
            C[N // 2, i - 1] = (delta**2 / 3.0) * (
                (i - 2) * f_fourier_coeff[N // 2, i - 2]
                + 4 * (i - 1) * f_fourier_coeff[N // 2, i - 1]
                + i * f_fourier_coeff[N // 2, i]
            )

            if i != 2:
                # n=0 mode with logs
                D[0, i - 1] = (delta**2 / 3.0) * (
                    (i - 2)
                    * np.log((i - 2) * delta)
                    * f_fourier_coeff[N // 2, i - 2]
                    + 4 * (i - 1)
                    * np.log((i - 1) * delta)
                    * f_fourier_coeff[N // 2, i - 1]
                    + i * np.log(i * delta)
                    * f_fourier_coeff[N // 2, i]
                )

        C[N // 2, 0] = (delta**2 / 2.0) * f_fourier_coeff[N // 2, 1]
        D[0, 1] = (delta**2 / 3.0) * (
            4 * np.log(delta) * f_fourier_coeff[N // 2, 1]
            + 2 * np.log(2 * delta) * f_fourier_coeff[N // 2, 2]
        )
        D[0, 0] = (delta**2 / 2.0) * (
            (M - 2) * np.log((M - 2) * delta) * f_fourier_coeff[N // 2, M - 2]
            + (M - 1) * np.log((M - 1) * delta) * f_fourier_coeff[N // 2, M - 1]
        )

    else:
        raise ValueError("Unknown quad_rule; must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
