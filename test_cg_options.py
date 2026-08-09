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

print(f"{'Mesh':<15} | {'Weight':<10} | {'Warm Start':<10} | {'L2 Error':<12} | {'Time (s)':<10}")
print("-" * 65)

for mesh in mesh_types:
    method = make_radial_method(f'{mesh}-NUFFT', f'{mesh} / NUFFT', rad_unif=1, azu_unif=1)
    method['mesh_kind'] = mesh
    method['use_nudft'] = False
    
    for weight in weights:
        for warm in warm_starts:
            nonuniform.WEIGHT_TYPE = weight
            nonuniform.WARM_START = warm
            
            try:
                res = run_case_radial(N, M, method, mute=True)
                err = res['L2_rel']
                rt = res['runtime']
                print(f"{mesh:<15} | {weight:<10} | {warm:<10} | {err:<12.2e} | {rt:<10.4f}")
            except Exception as e:
                print(f"{mesh:<15} | {weight:<10} | {warm:<10} | {'FAILED':<12} | {str(e)[:10]}")
    print("-" * 65)
