import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import sympy as sp
from Poisson_Solver.cpu_solver.fourier.nonuniform import _invert_nudft, _invert_nufft_block_cgls_shared
from Poisson_Solver.cpu_solver.fourier.uniform import compute_fourier_coeff_unif
from Tests.paper.helpers import generate_adapted_clustered_azimuthal

# Test pure 1D Fourier interpolation accuracy on non-uniform vs uniform grid:
theta_0 = np.pi
gamma = 0.90
th_sym = sp.symbols('th', real=True)
h_sym = 1.0 / (1.0 - gamma * sp.cos(th_sym - theta_0))
h_func = sp.lambdify(th_sym, h_sym, "numpy")

print("--- 1D Angular Fourier Recovery Error ---")
for N in [32, 48, 64, 96, 128]:
    # Uniform
    th_un = np.linspace(0, 2*np.pi, N, endpoint=False)
    f_un = np.repeat(h_func(th_un)[:, None], 2, axis=1)  # (N, 2)
    c_un = compute_fourier_coeff_unif(f_un)
    
    # Non-uniform
    th_ad = generate_adapted_clustered_azimuthal(N, cluster_strength=0.50, center=theta_0)
    f_ad = np.repeat(h_func(th_ad)[:, None], 2, axis=1)
    c_nudft = _invert_nudft(th_ad, f_ad)
    c_nufft = _invert_nufft_block_cgls_shared(th_ad, f_ad, tol=1e-12, maxiter=200)
    
    # Check reconstruction at test points
    th_test = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    f_exact = h_func(th_test)
    
    k_vec = np.arange(-N//2, N//2)
    # Reconstruct from nudft
    f_rec_nudft = np.exp(1j * np.outer(th_test, k_vec)) @ c_nudft[:, 0]
    err_nudft = np.linalg.norm(f_rec_nudft.flatten() - f_exact) / np.linalg.norm(f_exact)

    # Reconstruct from nufft
    f_rec_nufft = np.exp(1j * np.outer(th_test, k_vec)) @ c_nufft[:, 0]
    err_nufft = np.linalg.norm(f_rec_nufft.flatten() - f_exact) / np.linalg.norm(f_exact)
    
    # Reconstruct from uniform
    c_un_core = c_un[:N, 0].copy()
    c_un_core[0] *= 2.0  # un-halve endpoint
    f_rec_unif = np.exp(1j * np.outer(th_test, k_vec)) @ c_un_core
    err_unif = np.linalg.norm(f_rec_unif.flatten() - f_exact) / np.linalg.norm(f_exact)
    
    print(f"N={N:3d} | Uniform FFT err: {err_unif:.3e} | NUDFT err: {err_nudft:.3e} | NUFFT err: {err_nufft:.3e}")
