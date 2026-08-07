import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import sympy as sp
import numpy as np
from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_cartesian_grid_on_disk,
    generate_grid_values,
    generate_clustered_azimuthal,
    generate_stratified_rand_azimuthal,
    compute_zero_mode
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver

# Construct smooth Gaussian peak on disk centered at (x0, y0)
R = 1.0
r0 = 0.6
theta0 = np.pi / 3  # 60 degrees
x0 = r0 * np.cos(theta0)
y0 = r0 * np.sin(theta0)
gamma = 30.0

x, y = sp.symbols('x y', real=True)
u_sym = (R**2 - x**2 - y**2) * sp.exp(-gamma * ((x - x0)**2 + (y - y0)**2))
f_sym = sp.diff(u_sym, x, 2) + sp.diff(u_sym, y, 2)

u_func = sp.lambdify((x, y), u_sym, "numpy")
f_func = sp.lambdify((x, y), f_sym, "numpy")

print("SymPy differentiation complete. Testing solver...")

# Test 1: Uniform grid
M = 64
for N in [32, 64, 128, 256, 512]:
    iRadius = generate_uniform_radial(M, R)
    iAngle = generate_uniform_azimuthal(N)
    x_c, y_c = generate_cartesian_grid_on_disk(iAngle, iRadius)
    f_vals = generate_grid_values(f_func, x_c, y_c)
    u_true = generate_grid_values(u_func, x_c, y_c)
    g_vals = generate_grid_values(u_func, x_c[:, -1], y_c[:, -1])
    
    u_approx = poisson_solver(
        f_vals, g_vals, np.array([]), N, M, iRadius, iAngle, R,
        quad_rule=1, BC_choice=1, rad_unif=1, azu_unif=2
    )
    _, _, _, l2_rel = compute_error_metrics(u_approx, u_true, iRadius, iAngle)
    print(f"Uniform FFT N={N:4d}, M={M:2d}: L2_rel = {l2_rel:.3e}")

# Test 2: Adapted clustered grid
for N in [32, 48, 64]:
    iRadius = generate_uniform_radial(M, R)
    # Cluster angles around theta0
    iAngle = generate_clustered_azimuthal(N, cluster_strength=2.0, center=theta0)
    x_c, y_c = generate_cartesian_grid_on_disk(iAngle, iRadius)
    f_vals = generate_grid_values(f_func, x_c, y_c)
    u_true = generate_grid_values(u_func, x_c, y_c)
    g_vals = generate_grid_values(u_func, x_c[:, -1], y_c[:, -1])
    
    u_approx = poisson_solver(
        f_vals, g_vals, np.array([]), N, M, iRadius, iAngle, R,
        quad_rule=1, BC_choice=1, rad_unif=1, azu_unif=1, use_nudft_angular=True
    )
    _, _, _, l2_rel = compute_error_metrics(u_approx, u_true, iRadius, iAngle)
    print(f"Adapted NUDFT N={N:4d}, M={M:2d}: L2_rel = {l2_rel:.3e}")
