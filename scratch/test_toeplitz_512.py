import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_fft
from Poisson_Solver.grids import generate_jittered_azimuthal
from Poisson_Solver.cpu_solver.fourier.nonuniform import (
    _wrap_angles,
    _compute_fft_kde_weights,
    _nufft_adjoint,
)

# Test Toeplitz baseline vs opt at N=512
N = 512
M = 512
theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.25)
f_arr = np.random.randn(N, M) + 1j * np.random.randn(N, M)

from scratch.test_optimizations_2_3_4 import toeplitz_baseline, toeplitz_optimized

t0 = time.perf_counter()
c_base = toeplitz_baseline(theta_j, f_arr)
t_base = time.perf_counter() - t0

t0 = time.perf_counter()
c_opt = toeplitz_optimized(theta_j, f_arr)
t_opt = time.perf_counter() - t0

max_diff = np.max(np.abs(c_base - c_opt))
speedup = t_base / t_opt

print(f"Toeplitz N={N}, M={M}: Base = {t_base:.3f}s | Opt = {t_opt:.3f}s | Speedup = {speedup:.2f}x | Max Diff = {max_diff:.2e}")
