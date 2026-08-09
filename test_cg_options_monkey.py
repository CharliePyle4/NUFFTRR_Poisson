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
warm_starts = ['zero', 'B', 'M_inv_B']

# Save original functions
orig_kde = nonuniform._compute_fft_kde_weights
orig_block_cg = nonuniform._block_cg

print(f"{'Mesh':<15} | {'Weight':<10} | {'Warm Start':<10} | {'L2 Error':<12} | {'Time (s)':<10}")
print("-" * 65)

for mesh in mesh_types:
    method = make_radial_method(f'{mesh}-NUFFT', f'{mesh} / NUFFT', rad_unif=1, azu_unif=1)
    method['mesh_kind'] = mesh
    method['use_nudft'] = False
    
    for weight in weights:
        for warm in warm_starts:
            # 1. Monkey-patch weighting
            if weight == 'kde':
                nonuniform._compute_fft_kde_weights = orig_kde
            elif weight == 'iterative':
                nonuniform._compute_fft_kde_weights = lambda theta: nonuniform._compute_iterative_dcf(theta, theta.size, iters=15)

            # 2. Monkey-patch warm starts
            def custom_block_cg(T_op, B, M_inv=None, tol=1e-8, maxiter=50):
                if warm == 'B':
                    X = B.copy()
                elif warm == 'M_inv_B' and M_inv is not None:
                    X = M_inv(B)
                else:
                    X = np.zeros_like(B)

                if warm == 'zero':
                    R = B.copy()
                else:
                    R = B - T_op(X)
                
                # Copy the rest of the original _block_cg logic exactly
                if M_inv is not None:
                    Z = M_inv(R)
                else:
                    Z = R.copy()

                P = Z.copy()
                gamma = np.vdot(R, Z).real
                norm_b_denom_sq = np.vdot(B, B).real + 1e-15
                tol2 = tol**2

                for _ in range(maxiter):
                    TP = T_op(P)
                    delta = np.vdot(P, TP).real
                    if delta <= 0.0 or np.isnan(delta):
                        break
                    
                    alpha = gamma / delta
                    X += alpha * P
                    R -= alpha * TP
                    
                    col_res_sq = np.einsum('ij,ij->i', R.real, R.real) + np.einsum('ij,ij->i', R.imag, R.imag)
                    if np.max(col_res_sq / norm_b_denom_sq) < tol2:
                        break
                    
                    if M_inv is not None:
                        Z = M_inv(R)
                    else:
                        Z = R.copy()
                        
                    gamma_new = np.vdot(R, Z).real
                    if gamma_new <= 0.0 or np.isnan(gamma_new):
                        break
                    
                    beta = gamma_new / gamma
                    P = Z + beta * P
                    gamma = gamma_new
                    
                return X
            
            nonuniform._block_cg = custom_block_cg
            
            try:
                res = run_case_radial(N, M, method, mute=True)
                err = res['L2_rel']
                rt = res['runtime']
                print(f"{mesh:<15} | {weight:<10} | {warm:<10} | {err:<12.2e} | {rt:<10.4f}")
            except Exception as e:
                print(f"{mesh:<15} | {weight:<10} | {warm:<10} | {'FAILED':<12} | {str(e)[:10]}")
    print("-" * 65)

# Restore originals
nonuniform._compute_fft_kde_weights = orig_kde
nonuniform._block_cg = orig_block_cg
