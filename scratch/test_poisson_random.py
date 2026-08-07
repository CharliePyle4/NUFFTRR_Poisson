import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from Tests.CPU.testing_helpers import _run_single_case, u_true, f_rhs, g_dirichlet, g_neumann

# Problem 1 setup
np.random.seed(42)
N = 64
M = 64

# Pure random grid
th_rand = np.sort(np.random.uniform(0, 2 * np.pi, N))

res_rand = _run_single_case(
    N=N, M=M,
    method={"name": "Rand-NUFFT", "label": "Rand / NUFFT", "azu_unif": 1, "use_nudft": False, "solver_azu_unif": 1},
    iAngle=th_rand,
    test_type="P1_Table1",
    bc_choice=1, quad_rule=2, R=1.0, mute=False
)

print(f"Problem 1 on Random Grid (N=64, M=64): L2 relative error = {res_rand['L2_rel']:.3e}")
