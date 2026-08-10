import os, sys
import numpy as np
import pandas as pd

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

# Test 2D (N, M) pairs for Clustered NUDFT and Sine NUDFT across M in [32, 128, 512] and N in [32, 64, 128, 256, 512]
cand_conds = [1e-16, 1e-14, 1e-12, 1e-11, 1e-10, 1e-8]
methods = [
    ("Fixed-clustered-NUDFT", "clustered", True),
    ("Fixed-sine-NUDFT", "sine", True),
    ("Fixed-sine-NUFFT", "sine", False),
    ("Fixed-clustered-NUFFT", "clustered", False),
]

cand_lambdas_nufft = [1e-12, 1e-10, 1e-8, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]

print("Optimal 2D (N, M) parameter lookup:")
for name, kind, use_nudft in methods:
    print(f"\n--- {name} ---")
    method = dict(name=name, label=name.replace("-", " "), azu_unif=1, mesh_kind=kind, solver_azu_unif=1, use_nudft=use_nudft)
    grid_lambdas = cand_conds if use_nudft else cand_lambdas_nufft
    for M in [32, 128, 512]:
        for N in [32, 64, 128, 256, 512]:
            best_err = 1e9
            best_lam = None
            for lam in grid_lambdas:
                set_global_config(reg_param=lam, tol_nufft=1e-10, maxiter_nufft=200)
                res = run_case(N, M, method, mute=True)
                err = res['L2_rel']
                if err < best_err:
                    best_err = err
                    best_lam = lam
            print(f"  (N={N:3d}, M={M:3d}): best={best_lam:.1e} -> error={best_err:.3e}")
