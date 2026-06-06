# ---------------------------------------------------------
# Fourier Coefficient Computation — uniform (azu_unif == 2)
# ---------------------------------------------------------
def compute_fourier_coeff_unif(f_values: np.ndarray) -> np.ndarray:
    f_values = np.asarray(f_values)
    N    = f_values.shape[0]
    half = N // 2

    if _is_matrix(f_values):
        f_shift  = np.roll(f_values, 1, axis=0)
        fft_vals = np.fft.fft(f_shift, axis=0) / N
        coeff    = np.vstack([fft_vals[half:N, :], fft_vals[0:half + 1, :]])
        coeff[0, :] /= 2.0
        coeff[N, :] /= 2.0
    else:
        f_shift  = np.roll(f_values, 1)
        fft_vals = np.fft.fft(f_shift) / N
        coeff    = np.hstack([fft_vals[half:N], fft_vals[0:half + 1]])
        coeff[0] /= 2.0
        coeff[N] /= 2.0

    return coeff
