import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from Tests.paper.helpers import (
    get_azimuthal_benchmark_problem,
    generate_adapted_clustered_azimuthal,
    run_benchmark_case
)

# Test with k=4, gamma=0.65
problem = get_azimuthal_benchmark_problem(R=1.0, theta_0=np.pi, k_mode=4, gamma=0.65)

print("Checking NUFFT and NUDFT with k=4, gamma=0.65, cluster=0.45:")
for N in [16, 24, 32, 48, 64, 128]:
    theta_ad = generate_adapted_clustered_azimuthal(N, cluster_strength=0.45, center=np.pi)
    theta_un = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # NUFFT
    res_nufft = run_benchmark_case(N, 64, 1, theta_ad, problem, quad_rule=2, use_nudft=False)
    # NUDFT
    res_nudft = run_benchmark_case(N, 64, 1, theta_ad, problem, quad_rule=2, use_nudft=True)
    # Uniform
    res_unif = run_benchmark_case(N, 64, 2, theta_un, problem, quad_rule=2, use_nudft=False)
    
    print(f"N={N:3d} | NUFFT: {res_nufft['L2_rel']:.3e} | NUDFT: {res_nudft['L2_rel']:.3e} | Uniform: {res_unif['L2_rel']:.3e}")
