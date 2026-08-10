import os, sys
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

fine_conds = [1e-16, 3e-16, 1e-15, 3e-15, 1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10]

for name, kind in [("Fixed-sine-NUDFT", "sine"), ("Fixed-clustered-NUDFT", "clustered")]:
    print(f"\n=== Fine cond search for {name} ===")
    method = dict(name=name, label=name.replace("-", " "), azu_unif=1, mesh_kind=kind, solver_azu_unif=1, use_nudft=True)
    for N in [64, 128]:
        best_err = 1e9
        best_c = None
        for c in fine_conds:
            set_global_config(reg_param=c)
            res = run_case(N, 512, method, mute=True)
            err = res['L2_rel']
            if err < best_err:
                best_err = err
                best_c = c
        print(f"  N={N:3d} (M=512): best cond = {best_c:.2e} -> error = {best_err:.4e}")
