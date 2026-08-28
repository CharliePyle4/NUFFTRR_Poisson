import sys
sys.path.insert(0, '.')
import numpy as np

from Poisson_Solver.cpu_solver.poisson_solver import poisson_solver
from Tests.CPUvsGPU.cpu_and_gpu_timing import (
    get_benchmark_problem,
    generate_jittered_azimuthal,
    generate_uniform_radial,
    generate_cartesian_grid_on_disk,
)

prob = get_benchmark_problem(R=1.0, k=2)

N, M = 64, 64
theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.25)
r_m = generate_uniform_radial(M, R=1.0)
x_mesh, y_mesh = generate_cartesian_grid_on_disk(theta_j, r_m)

f_vals = prob["f"](x_mesh, y_mesh)
g_vals = prob["g_dirichlet"](x_mesh[:, -1], y_mesh[:, -1])
u_exact = prob["u"](x_mesh, y_mesh)

# 1. NUDFT
u_nudft = poisson_solver(
    f_values=f_vals, g_values=g_vals, u_fourier_0=0.0,
    N=N, M=M, r_m=r_m, theta_j=theta_j, R=1.0,
    quad_rule=1, BC_choice=1, rad_unif=1, grid_type=2,
    use_nudft_angular=True, num_processors=1
)
err_nudft = np.max(np.abs(u_nudft - u_exact))

# 2. Toeplitz
u_toep = poisson_solver(
    f_values=f_vals, g_values=g_vals, u_fourier_0=0.0,
    N=N, M=M, r_m=r_m, theta_j=theta_j, R=1.0,
    quad_rule=1, BC_choice=1, rad_unif=1, grid_type=2,
    use_nudft_angular=False, num_processors=1
)
err_toep = np.max(np.abs(u_toep - u_exact))

# 3. PCGLS (grid_type=3)
u_pcgls = poisson_solver(
    f_values=f_vals, g_values=g_vals, u_fourier_0=0.0,
    N=N, M=M, r_m=r_m, theta_j=theta_j, R=1.0,
    quad_rule=1, BC_choice=1, rad_unif=1, grid_type=3,
    use_nudft_angular=False, num_processors=1
)
err_pcgls = np.max(np.abs(u_pcgls - u_exact))

print("ALL SOLVERS END-TO-END VALIDATION (N=64, M=64):")
print(f"  NUDFT Error:    {err_nudft:.4e}")
print(f"  Toeplitz Error: {err_toep:.4e}")
print(f"  PCGLS Error:    {err_pcgls:.4e}")

# Check difference between solvers
diff_toep_pcgls = np.max(np.abs(u_toep - u_pcgls))
print(f"  Difference (Toeplitz vs PCGLS): {diff_toep_pcgls:.4e}")
