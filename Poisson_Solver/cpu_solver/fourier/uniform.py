import numpy as np
from .nonuniform import _is_matrix

import pyfftw
import multiprocessing
import pyfftw.interfaces.numpy_fft as fftw_fft
pyfftw.interfaces.cache.enable()

# ---------------------------------------------------------
# Fourier Coefficient Computation — uniform (azu_unif == 2)
# ---------------------------------------------------------
def compute_fourier_coeff_unif(f_values: np.ndarray) -> np.ndarray:
    f_values = np.asarray(f_values)
    N    = f_values.shape[0]
    half = N // 2
    
    n_threads = multiprocessing.cpu_count()

    if _is_matrix(f_values):
        fft_vals = fftw_fft.fft(f_values, axis=0, threads=n_threads) / N
        coeff    = np.vstack([fft_vals[half:N, :], fft_vals[0:half + 1, :]])
        coeff[0, :] /= 2.0
        coeff[N, :] /= 2.0
    else:
        fft_vals = fftw_fft.fft(f_values, threads=n_threads) / N
        coeff    = np.hstack([fft_vals[half:N], fft_vals[0:half + 1]])
        coeff[0] /= 2.0
        coeff[N] /= 2.0

    return coeff
