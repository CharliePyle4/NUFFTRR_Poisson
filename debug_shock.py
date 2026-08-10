import os, sys
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

method = dict(name="Fixed-shock_layer-NUDFT", label="Fixed Shock Layer / NUDFT", azu_unif=1, mesh_kind="shock_layer", solver_azu_unif=1, use_nudft=True)

for cond in [1e-16, 1e-14, 1e-12, 1e-11, 1e-10, 1e-8]:
    set_global_config(reg_param=cond)
    res32 = run_case(32, 128, method, mute=False)
    res64 = run_case(64, 128, method, mute=False)
    res128 = run_case(128, 128, method, mute=False)
    print(f"cond={cond:.1e} | N=32: {res32['L2_rel']:.3e} | N=64: {res64['L2_rel']:.3e} | N=128: {res128['L2_rel']:.3e}")
