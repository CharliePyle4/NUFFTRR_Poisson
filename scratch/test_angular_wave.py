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
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver

R = 1.0
theta0 = np.pi / 2

x, y = sp.symbols('x y', real=True)
r_sq = x**2 + y**2
# High azimuthal frequency wave packet with localized amplitude
u_sym = (R**2 - r_sq) * (x**5 - 10*x**3*y**2 + 5*x*y**4) * sp.exp(2.5 * y)

f_sym = sp.diff(u_sym, x, 2) + sp.diff(u_sym, y, 2)

u_func = sp.lambdify((x, y), u_sym, "numpy")
f_func = sp.lambdify((x, y), f_sym, "numpy")

def generate_adapted_angles(N, alpha=0.6, center=theta0):
    s = np.linspace(0.0, 2*np.pi, N, endpoint=False)
    theta = s - alpha * np.sin(s - center)
    theta = np.mod(theta, 2*np.pi)
    return np.sort(theta)

M = 256
print(f"Running with M={M} (Simpson Rule)...")
for N in [16, 24, 32, 48, 64, 128]:
    iRadius = generate_uniform_radial(M, R)
    iAngle_ad = generate_adapted_angles(N, alpha=0.7, center=theta0)
    iAngle_un = generate_uniform_azimuthal(N)
    
    # Adapted solve
    x_ad, y_ad = generate_cartesian_grid_on_disk(iAngle_ad, iRadius)
    f_ad = generate_grid_values(f_func, x_ad, y_ad)
    u_true_ad = generate_grid_values(u_func, x_ad, y_ad)
    g_ad = generate_grid_values(u_func, x_ad[:, -1], y_ad[:, -1])
    
    u_app_ad = poisson_solver(
        f_ad, g_ad, np.array([]), N, M, iRadius, iAngle_ad, R,
        quad_rule=2, BC_choice=1, rad_unif=1, azu_unif=1, use_nudft_angular=False
    )
    _, _, _, l2_ad = compute_error_metrics(u_app_ad, u_true_ad, iRadius, iAngle_ad)
    
    # Uniform solve
    x_un, y_un = generate_cartesian_grid_on_disk(iAngle_un, iRadius)
    f_un = generate_grid_values(f_func, x_un, y_un)
    u_true_un = generate_grid_values(u_func, x_un, y_un)
    g_un = generate_grid_values(u_func, x_un[:, -1], y_un[:, -1])
    
    u_app_un = poisson_solver(
        f_un, g_un, np.array([]), N, M, iRadius, iAngle_un, R,
        quad_rule=2, BC_choice=1, rad_unif=1, azu_unif=2
    )
    _, _, _, l2_un = compute_error_metrics(u_app_un, u_true_un, iRadius, iAngle_un)
    
    print(f"N={N:3d} | Adapted NUFFT L2 = {l2_ad:.3e} | Uniform FFT L2 = {l2_un:.3e}")
