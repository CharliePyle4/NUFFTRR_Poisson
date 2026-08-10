import os, sys
import numpy as np

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Tests.CPU.testing_helpers import run_case, set_global_config

print("=== 1. Testing Multipole NUDFT with cond=1e-12 vs 1e-16 at N=256, 512 (M=128) ===")
method_mp = dict(name="Fixed-multipole-NUDFT", label="Fixed Multi-Pole / NUDFT", azu_unif=1, mesh_kind="multipole", solver_azu_unif=1, use_nudft=True)
for c in [1e-16, 1e-14, 1e-12, 1e-11, 1e-10]:
    set_global_config(reg_param=c)
    res256 = run_case(256, 128, method_mp, mute=True)
    res512 = run_case(512, 128, method_mp, mute=True)
    print(f"  cond={c:.1e} -> N=256: {res256['L2_rel']:.3e} | N=512: {res512['L2_rel']:.3e}")

print("\n=== 2. Testing Harmonic Warped NUDFT with cond=1e-12 vs 1e-16 at N=256, 512 (M=128) ===")
method_wp = dict(name="Fixed-warped-NUDFT", label="Fixed Harmonic Warped / NUDFT", azu_unif=1, mesh_kind="warped", solver_azu_unif=1, use_nudft=True)
for c in [1e-16, 1e-14, 1e-12, 1e-11, 1e-10]:
    set_global_config(reg_param=c)
    res256 = run_case(256, 128, method_wp, mute=True)
    res512 = run_case(512, 128, method_wp, mute=True)
    print(f"  cond={c:.1e} -> N=256: {res256['L2_rel']:.3e} | N=512: {res512['L2_rel']:.3e}")
