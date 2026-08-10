import os, sys
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

print("=== Testing Shock Layer grid with gamma=0.85 and cond=1e-12 ===")
method_sl_nudft = dict(name="Fixed-shock_layer-NUDFT", label="Fixed Shock Layer / NUDFT", azu_unif=1, mesh_kind="shock_layer", solver_azu_unif=1, use_nudft=True)
method_sl_nufft = dict(name="Fixed-shock_layer-NUFFT", label="Fixed Shock Layer / NUFFT", azu_unif=1, mesh_kind="shock_layer", solver_azu_unif=1, use_nudft=False)

set_global_config(reg_param=1e-12)
for N in [32, 64, 128, 256, 512]:
    res_nudft = run_case(N, 128, method_sl_nudft, mute=True)
    res_nufft = run_case(N, 128, method_sl_nufft, mute=True)
    print(f"N={N:3d} (M=128) | NUDFT: {res_nudft['L2_rel']:.3e} | NUFFT: {res_nufft['L2_rel']:.3e}")
