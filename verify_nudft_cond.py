import numpy as np
from scipy.linalg import lstsq
import os, sys

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

print("=== SINE NUDFT at N=128, M=32 with different lambda (rcond) ===")
method_sine_nudft = dict(name="Fixed-sine-NUDFT", label="Fixed sine / NUDFT", azu_unif=1, mesh_kind="sine", solver_azu_unif=1, use_nudft=True)
for cond in [1e-20, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8]:
    set_global_config(reg_param=cond)
    res = run_case(128, 32, method_sine_nudft, mute=False)
    print(f"  cond = {cond:.1e} -> error = {res['L2_rel']:.4e}")

print("\n=== CLUSTERED NUDFT at N=128, M=32 with different lambda (rcond) ===")
method_clust_nudft = dict(name="Fixed-clustered-NUDFT", label="Fixed clustered / NUDFT", azu_unif=1, mesh_kind="clustered", solver_azu_unif=1, use_nudft=True)
for cond in [1e-20, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8]:
    set_global_config(reg_param=cond)
    res = run_case(128, 32, method_clust_nudft, mute=False)
    print(f"  cond = {cond:.1e} -> error = {res['L2_rel']:.4e}")
