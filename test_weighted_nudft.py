import os, sys
import numpy as np
from scipy.linalg import lstsq

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal

for kind in ["sine", "clustered"]:
    print(f"\n=================== {kind.upper()} MESH ===================")
    for N in [32, 64, 128, 256, 512]:
        theta = generate_fixed_nonuniform_azimuthal(N, kind=kind)
        
        # Compute Voronoi / trapezoidal weights
        dtheta = np.zeros(N)
        theta_sorted_idx = np.argsort(theta)
        th_s = theta[theta_sorted_idx]
        th_ext = np.concatenate(([th_s[-1] - 2*np.pi], th_s, [th_s[0] + 2*np.pi]))
        w_s = 0.5 * (th_ext[2:] - th_ext[:-2])
        w = np.zeros(N)
        w[theta_sorted_idx] = w_s
        
        k = np.arange(-N // 2, N // 2, dtype=float)
        A = np.exp(1j * np.outer(theta, k))
        
        # Test function f(theta) = cos(theta) + sin(2*theta) (smooth Fourier modes)
        c_true = np.zeros(N, dtype=complex)
        c_true[N//2 + 1] = 0.5
        c_true[N//2 - 1] = 0.5
        c_true[N//2 + 2] = -0.5j
        c_true[N//2 - 2] = 0.5j
        f = A @ c_true
        
        # Unweighted lstsq
        c_unweighted = lstsq(A, f, cond=1e-12)[0]
        err_unw = np.linalg.norm(c_unweighted - c_true) / np.linalg.norm(c_true)
        
        # Weighted lstsq
        W_sqrt = np.sqrt(w)[:, None]
        c_weighted = lstsq(W_sqrt * A, W_sqrt * f, cond=1e-12)[0]
        err_w = np.linalg.norm(c_weighted - c_true) / np.linalg.norm(c_true)
        
        # Trapezoidal adjoint (quadrature analysis)
        c_quad = (A.conj().T @ (w[:, None] * f)) / (2*np.pi)
        err_quad = np.linalg.norm(c_quad - c_true) / np.linalg.norm(c_true)
        
        # Condition numbers
        s_unw = np.linalg.svd(A, compute_uv=False)
        s_w = np.linalg.svd(W_sqrt * A, compute_uv=False)
        cond_unw = s_unw[0] / s_unw[-1]
        cond_w = s_w[0] / s_w[-1]
        
        print(f"N={N:3d} | unw_err={err_unw:.2e} (cond={cond_unw:.1e}) | weighted_err={err_w:.2e} (cond={cond_w:.1e}) | quad_err={err_quad:.2e}")
