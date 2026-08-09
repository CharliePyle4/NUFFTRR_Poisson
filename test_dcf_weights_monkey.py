import sys
sys.path.append('c:/Users/charl/NUFFTRR_Poisson')
import time
import numpy as np

from Tests.CPU.testing_helpers import run_case_radial, make_radial_method, set_global_config
from Poisson_Solver.cpu_solver.fourier import nonuniform

# Global Settings
N = 32
M = 32
set_global_config(maxiter_nufft=200, tol_nufft=1e-10, reg_param=1e-6)

mesh_types = ['uniform', 'jittered', 'sine', 'clustered']
weights = ['kde', 'iterative']

# Save original functions
orig_kde = nonuniform._compute_fft_kde_weights

# Define the iterative DCF so we don't have to edit nonuniform.py
def _compute_iterative_dcf(theta_j: np.ndarray, N_modes: int, iters: int = 15, eps: float = 1e-12) -> np.ndarray:
    x_wrapped = nonuniform._wrap_angles(theta_j)
    w = nonuniform._get_density_weights(theta_j)
    
    for _ in range(iters):
        u = nonuniform._nufft_adjoint(x_wrapped, w, N_modes=N_modes, eps=eps)
        v = nonuniform._nufft_forward(x_wrapped, u, eps=eps)
        w = w / (v.real + 1e-14)
        
    return w

print(f"{'Mesh':<15} | {'Weight':<10} | {'L2 Error':<12} | {'Time (s)':<10}")
print("-" * 55)

for mesh in mesh_types:
    method = make_radial_method(f'{mesh}-NUFFT', f'{mesh} / NUFFT', rad_unif=1, azu_unif=1)
    method['mesh_kind'] = mesh
    method['use_nudft'] = False
    
    for weight in weights:
        # 1. Monkey-patch weighting
        if weight == 'kde':
            nonuniform._compute_fft_kde_weights = orig_kde
        elif weight == 'iterative':
            nonuniform._compute_fft_kde_weights = lambda theta: _compute_iterative_dcf(theta, theta.size, iters=15)
        
        try:
            res = run_case_radial(N, M, method, mute=True)
            err = res['L2_rel']
            rt = res['runtime']
            print(f"{mesh:<15} | {weight:<10} | {err:<12.2e} | {rt:<10.4f}")
        except Exception as e:
            print(f"{mesh:<15} | {weight:<10} | {'FAILED':<12} | {str(e)[:10]}")
    print("-" * 55)

# Restore originals
nonuniform._compute_fft_kde_weights = orig_kde
