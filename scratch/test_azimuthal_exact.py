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

# Consider u(r, theta) = r^2 * (R^2 - r^2) * h(theta)
# where h(theta) is a localized angular profile:
# h(theta) = 1 / (1 + beta * cos(theta - theta0)) with beta = 0.8
# In polar coordinates:
# Δu = (1/r) d/dr(r du/dr) + (1/r^2) d^2u/dth^2
# d/dr [ r * d/dr (r^2(R^2 - r^2)) ] = d/dr [ r * (2r R^2 - 4r^3) ] = d/dr [ 2r^2 R^2 - 4r^4 ] = 4r R^2 - 16r^3
# (1/r) * (4r R^2 - 16r^3) = 4 R^2 - 16 r^2.
# And (1/r^2) d^2u/dth^2 = (R^2 - r^2) * d^2 h / dth^2 !
# Notice that at r -> 0, (R^2 - r^2) * d^2 h / dth^2 -> R^2 * d^2 h / dth^2, which is bounded and smooth!

beta = 0.8
th_sym = sp.symbols('th', real=True)
h_sym = 1 / (1 + beta * sp.cos(th_sym - theta0))
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
    # Δu = (4*R^2 - 16*r^2) * h(theta) + (R^2 - r^2) * h''(theta)
    return (4*R**2 - 16*rc**2) * h_func(thc) + (R**2 - rc**2) * h_d2_func(thc)

def generate_adapted_angles(N, alpha=0.7, center=theta0):
    s = np.linspace(0.0, 2*np.pi, N, endpoint=False)
    # Bijective mapping concentrating points near center
    theta = s - alpha * np.sin(s - center)
    theta = np.mod(theta, 2*np.pi)
    return np.sort(theta)

M = 64
print(f"Testing Localized Angular Function with beta={beta}, M={M} (Simpson Rule):")
print("-" * 75)
print(f"{'N':>5} | {'Adapted NUFFT L2':>18} | {'Uniform FFT L2':>16} | {'Error Ratio':>12}")
print("-" * 75)

for N in [16, 24, 32, 48, 64, 96, 128, 256, 512]:
    iRadius = generate_uniform_radial(M, R)
    iAngle_ad = generate_adapted_angles(N, alpha=0.72, center=theta0)
    iAngle_un = generate_uniform_azimuthal(N)
    
    # Adapted solve (NUFFT)
    x_ad, y_ad = generate_cartesian_grid_on_disk(iAngle_ad, iRadius)
    f_ad = f_rhs(x_ad, y_ad)
    u_true_ad = u_true(x_ad, y_ad)
    g_ad = u_true(x_ad[:, -1], y_ad[:, -1])
    
    u_app_ad = poisson_solver(
        f_ad, g_ad, np.array([]), N, M, iRadius, iAngle_ad, R,
        quad_rule=2, BC_choice=1, rad_unif=1, azu_unif=1, use_nudft_angular=False
    )
    _, _, _, l2_ad = compute_error_metrics(u_app_ad, u_true_ad, iRadius, iAngle_ad)
    
    # Uniform solve (FFT)
    x_un, y_un = generate_cartesian_grid_on_disk(iAngle_un, iRadius)
    f_un = f_rhs(x_un, y_un)
    u_true_un = u_true(x_un, y_un)
    g_un = u_true(x_un[:, -1], y_un[:, -1])
    
    u_app_un = poisson_solver(
        f_un, g_un, np.array([]), N, M, iRadius, iAngle_un, R,
        quad_rule=2, BC_choice=1, rad_unif=1, azu_unif=2
    )
    _, _, _, l2_un = compute_error_metrics(u_app_un, u_true_un, iRadius, iAngle_un)
    
    ratio = l2_un / max(l2_ad, 1e-16)
    print(f"{N:5d} | {l2_ad:18.3e} | {l2_un:16.3e} | {ratio:11.1f}x")
