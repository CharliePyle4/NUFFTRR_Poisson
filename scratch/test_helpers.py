import sys
import os
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from Tests.paper.helpers import (
    get_azimuthal_benchmark_problem,
    generate_adapted_clustered_azimuthal,
    run_benchmark_case,
    run_crankup_study
)

print("Constructing benchmark problem...")
problem = get_azimuthal_benchmark_problem(R=1.0, theta_0=np.pi, kappa=25.0)

print("Testing adapted solve (N=32, M=64)...")
theta_adapt = generate_adapted_clustered_azimuthal(32, cluster_strength=2.5, center=np.pi)
res_adapt = run_benchmark_case(
    N=32, M=64, azu_unif=1, theta_j=theta_adapt,
    problem=problem, bc_choice=1, quad_rule=1, use_nudft=False
)
print(f"Adapted NUFFT Result: L2_rel = {res_adapt['L2_rel']:.3e}, time = {res_adapt['runtime']:.4f}s")

print("Testing uniform solve (N=32, M=64)...")
theta_unif = np.linspace(0, 2*np.pi, 32, endpoint=False)
res_unif_32 = run_benchmark_case(
    N=32, M=64, azu_unif=2, theta_j=theta_unif,
    problem=problem, bc_choice=1, quad_rule=1
)
print(f"Uniform FFT Result (N=32): L2_rel = {res_unif_32['L2_rel']:.3e}, time = {res_unif_32['runtime']:.4f}s")

print("Testing uniform solve (N=256, M=64)...")
theta_unif_256 = np.linspace(0, 2*np.pi, 256, endpoint=False)
res_unif_256 = run_benchmark_case(
    N=256, M=64, azu_unif=2, theta_j=theta_unif_256,
    problem=problem, bc_choice=1, quad_rule=1
)
print(f"Uniform FFT Result (N=256): L2_rel = {res_unif_256['L2_rel']:.3e}, time = {res_unif_256['runtime']:.4f}s")

print("All test runs completed successfully!")
