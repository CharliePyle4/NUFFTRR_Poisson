import cupy as cp
from .nonuniform import _is_matrix

# ---------------------------------------------------------
# Fourier Coefficient Computation — uniform (grid_type == 1)
# ---------------------------------------------------------
def compute_fourier_coeff_unif(f_values: cp.ndarray, **kwargs) -> cp.ndarray:
    f_values = cp.asarray(f_values)
    N = f_values.shape[0]
    half = N // 2

    if _is_matrix(f_values):
        fft_vals = cp.fft.fft(f_values, axis=0) / N
        coeff = cp.vstack([fft_vals[half:N, :], fft_vals[0:half + 1, :]])
        coeff[0, :] /= 2.0
        coeff[N, :] /= 2.0
    else:
        fft_vals = cp.fft.fft(f_values) / N
        coeff = cp.hstack([fft_vals[half:N], fft_vals[0:half + 1]])
        coeff[0] /= 2.0
        coeff[N] /= 2.0

    return coeff
