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

print("Checking Toeplitz vs PCGLS vs NUDFT accuracy across grid sizes:")
for N, M in [(32, 32), (32, 1024), (64, 64), (64, 1024), (128, 128), (256, 256)]:
    theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.25)
    r_m = generate_uniform_radial(M, R=1.0)
    x_mesh, y_mesh = generate_cartesian_grid_on_disk(theta_j, r_m)

    f_vals = prob["f"](x_mesh, y_mesh)
    g_vals = prob["g_dirichlet"](x_mesh[:, -1], y_mesh[:, -1])
    u_exact = prob["u"](x_mesh, y_mesh)

    u_toep = poisson_solver(
        f_values=f_vals, g_values=g_vals, u_fourier_0=0.0,
        N=N, M=M, r_m=r_m, theta_j=theta_j, R=1.0,
        quad_rule=1, BC_choice=1, rad_unif=1, grid_type=2,
        use_nudft_angular=False, num_processors=1
    )
    err_toep = np.max(np.abs(u_toep - u_exact))

    u_pcgls = poisson_solver(
        f_values=f_vals, g_values=g_vals, u_fourier_0=0.0,
        N=N, M=M, r_m=r_m, theta_j=theta_j, R=1.0,
        quad_rule=1, BC_choice=1, rad_unif=1, grid_type=3,
        use_nudft_angular=False, num_processors=1
    )
    err_pcgls = np.max(np.abs(u_pcgls - u_exact))

    print(f"N={N:4d}, M={M:4d} | Toeplitz Error: {err_toep:.4e} | PCGLS Error: {err_pcgls:.4e} | Rel Diff: {abs(err_toep - err_pcgls):.2e}")
