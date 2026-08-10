import os, sys
import numpy as np
from scipy.linalg import lstsq, pinv

repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
os.chdir(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal
from Tests.CPU.testing_helpers import run_case, set_global_config

print("=== Investigating NUDFT inversion on SINE mesh ===")
for N in [32, 64, 128, 256, 512]:
    theta = generate_fixed_nonuniform_azimuthal(N, kind="sine")
    k = np.arange(-N // 2, N // 2, dtype=float)
    A = np.exp(1j * np.outer(theta, k))
    s = np.linalg.svd(A, compute_uv=False)
    cond = s[0] / s[-1]
    print(f"N={N:3d} | cond(A)={cond:.2e} | s_max={s[0]:.2f} | s_min={s[-1]:.2e} | s_median={np.median(s):.2f}")

print("\n=== Investigating NUDFT inversion on CLUSTERED mesh ===")
for N in [32, 64, 128, 256, 512]:
    theta = generate_fixed_nonuniform_azimuthal(N, kind="clustered")
    k = np.arange(-N // 2, N // 2, dtype=float)
    A = np.exp(1j * np.outer(theta, k))
    s = np.linalg.svd(A, compute_uv=False)
    cond = s[0] / s[-1]
    print(f"N={N:3d} | cond(A)={cond:.2e} | s_max={s[0]:.2f} | s_min={s[-1]:.2e} | s_median={np.median(s):.2f}")
