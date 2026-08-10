import os, sys
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal
import Poisson_Solver.cpu_solver.fourier.nonuniform as nonunif

# Let's patch _block_cg to remove the early exit and print the iteration count
def _block_cg_clean(T_op, B, M_inv=None, tol=1e-12, maxiter=500):
    X = B.copy()
    R = B - T_op(X)

    if M_inv is not None:
        Z = M_inv(R)
    else:
        Z = R.copy()

    P = Z.copy()
    gamma = np.vdot(R, Z).real

    if gamma <= 0.0 or np.isnan(gamma):
        return X

    norm_b_sq = np.einsum('ij,ij->i', B.real, B.real) + np.einsum('ij,ij->i', B.imag, B.imag)
    norm_b_denom_sq = (np.sqrt(norm_b_sq) + 1e-14)**2
    tol2 = tol * tol

    col_res_sq = np.einsum('ij,ij->i', R.real, R.real) + np.einsum('ij,ij->i', R.imag, R.imag)
    if np.max(col_res_sq / norm_b_denom_sq) < tol2:
        return X

    for it in range(maxiter):
        TP = T_op(P)
        delta = np.vdot(P, TP).real
        if delta <= 0 or np.isnan(delta):
            break

        alpha = gamma / delta

        X += alpha * P
        R -= alpha * TP

        col_res_sq = np.einsum('ij,ij->i', R.real, R.real) + np.einsum('ij,ij->i', R.imag, R.imag)
        if np.max(col_res_sq / norm_b_denom_sq) < tol2:
            print(f"    -> CG converged in {it+1} iters (res = {np.sqrt(np.max(col_res_sq/norm_b_denom_sq)):.2e})")
            break

        if M_inv is not None:
            Z_new = M_inv(R)
        else:
            Z_new = R.copy()

        gamma_new = np.vdot(R, Z_new).real
        if gamma_new < 1e-28:
            break

        beta = gamma_new / (gamma + 1e-14)
        P *= beta
        P += Z_new
        gamma = gamma_new

    return X

nonunif._block_cg = _block_cg_clean

N = 128
for kind in ["warped", "multipole", "chebyshev"]:
    theta = generate_fixed_nonuniform_azimuthal(N, kind=kind)
    f = np.exp(np.sin(theta)) * np.cos(2*theta)
    
    print(f"\nGrid: {kind}")
    c_nudft = nonunif._invert_nudft(theta, f, reg_param=1e-12)
    c_nufft = nonunif._invert_nufft_block_cgls_shared(theta, f, tol=1e-12, maxiter=500, reg_param=1e-12)
    
    err = np.linalg.norm(c_nufft - c_nudft) / np.linalg.norm(c_nudft)
    print(f"  Error vs NUDFT: {err:.4e}")
