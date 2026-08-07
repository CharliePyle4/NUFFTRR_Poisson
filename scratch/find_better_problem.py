import os
import sys
repo_root = r"c:\Users\charl\NUFFTRR_Poisson"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import sympy as sp
from Tests.paper.helpers import (
    generate_adapted_clustered_azimuthal,
    run_benchmark_case
)

def test_problem(k_param, width, cluster_str):
    # Test wrapped Gaussian or localized bump:
    # h(theta) = exp(kappa * cos(theta - theta_0)) or 1 / (1 + gamma * cos(theta - theta_0)) with high gamma
    # Let's test with gamma = 0.95 or 0.98, or exp(kappa * cos(theta - pi))
    R = 1.0
    theta_0 = np.pi
    
    th_sym = sp.symbols('th', real=True)
    # sharp peak at theta_0:
    # h(theta) = 1 / (1 - gamma * cos(theta - theta_0)) or exp(kappa * (cos(theta - theta_0) - 1))
    gamma = 0.95
    h_sym = 1.0 / (1.0 - gamma * sp.cos(th_sym - theta_0))
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
        return (4.0 * R**2 - 16.0 * rc**2) * h_func(thc) + (R**2 - rc**2) * h_d2_func(thc)

    def g_dir(xc, yc):
        return u_true(xc, yc)
        
    problem = {
        "u": u_true,
        "f": f_rhs,
        "g_dirichlet": g_dir,
        "g_neumann": lambda xc, yc, R_val=R: -2.0 * (R_val**3) * h_func(np.arctan2(yc, xc)),
        "h_func": h_func,
        "R": R,
        "theta_0": theta_0
    }
    
    print(f"\n--- Testing Sharp Problem (gamma={gamma}, cluster_str={cluster_str}) ---")
    for N in [32, 48, 64, 96, 128]:
        th_ad = generate_adapted_clustered_azimuthal(N, cluster_strength=cluster_str, center=theta_0)
        th_un = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        res_nufft = run_benchmark_case(N, 64, 1, th_ad, problem, quad_rule=2, use_nudft=False)
        res_nudft = run_benchmark_case(N, 64, 1, th_ad, problem, quad_rule=2, use_nudft=True)
        res_unif = run_benchmark_case(N, 64, 2, th_un, problem, quad_rule=2, use_nudft=False)
        
        print(f"N={N:3d} | NUFFT: {res_nufft['L2_rel']:.3e} | NUDFT: {res_nudft['L2_rel']:.3e} | Uniform: {res_unif['L2_rel']:.3e} | Ratio(Unif/NUFFT): {res_unif['L2_rel']/res_nufft['L2_rel']:.2f}x")

if __name__ == "__main__":
    for c in [0.60, 0.70, 0.80]:
        test_problem(4, 0.1, c)
