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

# Localized angular wave packet on disk
R = 1.0
theta0 = np.pi
k_freq = 6
sigma = 0.35  # narrow angular width

# In polar coordinates:
# u(r, theta) = (R^2 - r^2) * r * cos(k*theta) * exp(-((theta - theta0)/sigma)^2)
# Let's write a periodic Gaussian (von Mises type): exp(kappa * (cos(theta - theta0) - 1))
kappa_val = 15.0

# Using Cartesian symbols:
x, y = sp.symbols('x y', real=True)
r_sq = x**2 + y**2
r = sp.sqrt(r_sq)

# von Mises localized angular peak:
# cos(theta - theta0) = -x / r when theta0 = pi
cos_diff = -x / sp.Piecewise((r, r > 1e-14), (1.0, True))
# High frequency oscillation: cos(k*theta)
# For k=6, we can use Chebyshev polynomial T_6(cos theta) or write directly
# Let's test with a smooth localized Cartesian function:
u_sym = (R**2 - r_sq) * sp.sin(12 * sp.atan2(y, x)) * sp.exp(kappa_val * (cos_diff - 1))

# Or simply in polar coordinates using finite difference or exact polar Laplacian:
# Δu = u_rr + (1/r) u_r + (1/r^2) u_theta_theta
r_sym, th_sym = sp.symbols('r th', real=True)
u_polar = (R**2 - r_sym**2) * r_sym**2 * sp.cos(k_freq * th_sym) * sp.exp(kappa_val * (sp.cos(th_sym - theta0) - 1))
f_polar = sp.diff(u_polar, r_sym, 2) + (1/r_sym)*sp.diff(u_polar, r_sym) + (1/r_sym**2)*sp.diff(u_polar, th_sym, 2)
f_polar = sp.simplify(f_polar)

u_polar_func = sp.lambdify((r_sym, th_sym), u_polar, "numpy")
f_polar_func = sp.lambdify((r_sym, th_sym), f_polar, "numpy")

def u_eval(xc, yc):
    rc = np.sqrt(xc**2 + yc**2)
    thc = np.arctan2(yc, xc)
    thc = np.mod(thc, 2*np.pi)
    return u_polar_func(rc, thc)

def f_eval(xc, yc):
    rc = np.sqrt(xc**2 + yc**2)
    thc = np.arctan2(yc, xc)
    thc = np.mod(thc, 2*np.pi)
    # At r=0, r^2 * ... / r^2 is smooth and regular
    res = f_polar_func(np.maximum(rc, 1e-14), thc)
    res[rc < 1e-14] = 0.0
    return res

def generate_adapted_angles(N, alpha=0.75, center=theta0):
    s = np.linspace(0.0, 2*np.pi, N, endpoint=False)
    theta = s - alpha * np.sin(s - center)
    theta = np.mod(theta, 2*np.pi)
    return np.sort(theta)

M = 64
print("Comparing Adapted vs Uniform on Localized Wave Packet:")
for N in [32, 48, 64, 96, 128, 192, 256, 384, 512]:
    iRadius = generate_uniform_radial(M, R)
    iAngle_ad = generate_adapted_angles(N, alpha=0.82, center=theta0)
    iAngle_un = generate_uniform_azimuthal(N)
    
    # Adapted solve
    x_ad, y_ad = generate_cartesian_grid_on_disk(iAngle_ad, iRadius)
    f_ad = f_eval(x_ad, y_ad)
    u_true_ad = u_eval(x_ad, y_ad)
    g_ad = u_eval(x_ad[:, -1], y_ad[:, -1])
    
    u_app_ad = poisson_solver(
        f_ad, g_ad, np.array([]), N, M, iRadius, iAngle_ad, R,
        quad_rule=1, BC_choice=1, rad_unif=1, azu_unif=1, use_nudft_angular=False
    )
    _, _, _, l2_ad = compute_error_metrics(u_app_ad, u_true_ad, iRadius, iAngle_ad)
    
    # Uniform solve
    x_un, y_un = generate_cartesian_grid_on_disk(iAngle_un, iRadius)
    f_un = f_eval(x_un, y_un)
    u_true_un = u_eval(x_un, y_un)
    g_un = u_eval(x_un[:, -1], y_un[:, -1])
    
    u_app_un = poisson_solver(
        f_un, g_un, np.array([]), N, M, iRadius, iAngle_un, R,
        quad_rule=1, BC_choice=1, rad_unif=1, azu_unif=2
    )
    _, _, _, l2_un = compute_error_metrics(u_app_un, u_true_un, iRadius, iAngle_un)
    
    print(f"N={N:4d} | Adapted NUFFT L2 = {l2_ad:.3e} | Uniform FFT L2 = {l2_un:.3e}")
