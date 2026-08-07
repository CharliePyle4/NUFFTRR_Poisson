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
theta0 = np.pi
k_mode = 8
gamma = 0.92

# u(r, theta) = r^2 * (R^2 - r^2) * cos(k*theta) / (1 - gamma * cos(theta - theta0))
th_sym = sp.symbols('th', real=True)
h_sym = sp.cos(k_mode * th_sym) / (1 - gamma * sp.cos(th_sym - theta0))
h_d2_sym = sp.diff(h_sym, th_sym, 2)

h_func = sp.lambdify(th_sym, h_sym, "numpy")
h_d2_func = sp.lambdify(th_sym, h_d2_sym, "numpy")

def u_true(xc, yc):
    rc = np.sqrt(xc**2 + yc**2)
    thc = np.arctan2(yc, xc)
    return rc**2 * (R**2 - rc**2) * h_func(thc)

def f_rhs(xc, yc):
    rc = np.sqrt(xc**2 + yc**2)
    thc = np.arctan2(yc, xc)
    return (4*R**2 - 16*rc**2) * h_func(thc) + (R**2 - rc**2) * h_d2_func(thc)

def generate_adapted_angles(N, alpha=0.6, center=theta0):
    s = np.linspace(0.0, 2*np.pi, N, endpoint=False)
    theta = s - alpha * np.sin(s - center)
    theta = np.mod(theta, 2*np.pi)
    return np.sort(theta)

M = 64
print(f"Testing k={k_mode}, gamma={gamma}, M={M}:")
print("-" * 65)
for N in [32, 64, 128, 256, 512]:
    iRadius = generate_uniform_radial(M, R)
    iAngle_un = generate_uniform_azimuthal(N)
    
    x_un, y_un = generate_cartesian_grid_on_disk(iAngle_un, iRadius)
    f_un = f_rhs(x_un, y_un)
    u_true_un = u_true(x_un, y_un)
    g_un = u_true(x_un[:, -1], y_un[:, -1])
    
    u_app_un = poisson_solver(
        f_un, g_un, np.array([]), N, M, iRadius, iAngle_un, R,
        quad_rule=2, BC_choice=1, rad_unif=1, azu_unif=2
    )
    _, _, _, l2_un = compute_error_metrics(u_app_un, u_true_un, iRadius, iAngle_un)
    print(f"Uniform FFT N={N:4d} | L2 Error = {l2_un:.4e}")
