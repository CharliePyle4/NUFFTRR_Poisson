import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import finufft
from Tests.paper.helpers import generate_adapted_clustered_azimuthal

def _wrap_angles(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi

def compute_barnett_greengard_dcf(theta: np.ndarray, num_iter: int = 3, eps: float = 1e-12) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    N = theta.size
    x_wrapped = np.ascontiguousarray(_wrap_angles(theta))
    
    # Initial Voronoi weights
    th_sort_idx = np.argsort(theta)
    th_sort = theta[th_sort_idx]
    th_ext = np.concatenate(([th_sort[-1] - 2.0 * np.pi], th_sort, [th_sort[0] + 2.0 * np.pi]))
    w_sort = (th_ext[2:] - th_ext[:-2]) / (4.0 * np.pi)
    w = np.zeros_like(theta)
    w[th_sort_idx] = w_sort
    
    # Pipe-Menke iteration
    for it in range(num_iter):
        fhat = finufft.nufft1d1(x_wrapped, np.ascontiguousarray(w, dtype=np.complex128), n_modes=N, isign=-1, eps=eps)
        fhat /= float(N)
        denom = finufft.nufft1d2(x_wrapped, np.ascontiguousarray(fhat), isign=+1, eps=eps).real
        denom = np.maximum(denom, 1e-12)
        w = w / denom
        w /= np.sum(w)
    return w

# Test on clustered angles
N = 64
theta = generate_adapted_clustered_azimuthal(N, cluster_strength=0.40, center=np.pi)
w_init = ((np.concatenate(([theta[-1] - 2*np.pi], theta, [theta[0] + 2*np.pi]))[2:] - np.concatenate(([theta[-1] - 2*np.pi], theta, [theta[0] + 2*np.pi]))[:-2])) / (4*np.pi)
w_bg = compute_barnett_greengard_dcf(theta, num_iter=3)

print("Initial Voronoi weights min/max:", np.min(w_init), np.max(w_init))
print("Barnett-Greengard DCF weights min/max:", np.min(w_bg), np.max(w_bg))
print("Sum of weights:", np.sum(w_bg))
