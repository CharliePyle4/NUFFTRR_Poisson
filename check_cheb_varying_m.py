import os, sys
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

# Let's test with tol_nufft=1e-14 and maxiter_nufft=500
set_global_config(reg_param=1e-12, tol_nufft=1e-14, maxiter_nufft=500)

method_nudft = dict(name="Fixed-chebyshev-NUDFT", label="Fixed Chebyshev / NUDFT", azu_unif=1, mesh_kind="chebyshev", solver_azu_unif=1, use_nudft=True)
method_nufft = dict(name="Fixed-chebyshev-NUFFT", label="Fixed Chebyshev / NUFFT", azu_unif=1, mesh_kind="chebyshev", solver_azu_unif=1, use_nudft=False)

print("=== Chebyshev at N=128 (Varying M) ===")
for M in [32, 64, 128, 256, 512]:
    r_nudft = run_case(128, M, method_nudft, mute=True)
    r_nufft = run_case(128, M, method_nufft, mute=True)
    print(f"  M={M:3d} | NUDFT: {r_nudft['L2_rel']:.3e} | NUFFT (maxiter=500): {r_nufft['L2_rel']:.3e}")
