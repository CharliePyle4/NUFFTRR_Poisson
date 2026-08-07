import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from Tests.paper.helpers import (
    get_azimuthal_benchmark_problem,
    run_unified_crankup_study,
    run_vary_N_study,
    run_vary_M_study,
    run_NxM_matrix_study,
    render_unified_table
)

problem = get_azimuthal_benchmark_problem(R=1.0, theta_0=np.pi, k_mode=4, gamma=0.65)

print("1. Running Unified Crankup Study...")
unified = run_unified_crankup_study(
    problem, N_adapt_list=[32, 48, 64], N_unif_ladder=[32, 64, 128, 256, 512],
    M_radial=64, cluster_strength=0.40, theta_0=np.pi, bc_choice=1, quad_rule=2
)
for r in unified:
    print(f"  {r['label']:<30} | L2_rel = {r['L2_rel']:.3e} | runtime = {r['runtime']:.4f}s")

print("\n2. Running Varying N Study...")
df_N = run_vary_N_study(
    problem, N_values=[16, 24, 32, 48, 64, 128], M_fixed=64,
    cluster_strength=0.40, theta_0=np.pi, bc_choice=1, quad_rule=2
)
print(df_N[["method", "N", "M", "L2_rel", "runtime"]].head(15))

print("\n3. Running Varying M Study...")
df_M = run_vary_M_study(
    problem, M_values=[16, 32, 64, 128], N_fixed=64,
    cluster_strength=0.40, theta_0=np.pi, quad_rules=[1, 2], bc_choices=[1, 2]
)
print(df_M[["config", "M", "L2_rel", "runtime"]].head(15))

print("\nAll pipeline functions tested successfully!")
