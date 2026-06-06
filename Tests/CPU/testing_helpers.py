import os
import sys
import time
import numpy as np
import pandas as pd
from IPython.display import display, HTML

from Poisson_Solver_Testing_2.solver import (
    generate_grid_values,
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_fixed_nonuniform_azimuthal,
    generate_nonuniform_azimuthal,
    generate_cartesian_grid_on_disk,
    compute_error_metrics,
    poisson_solver,
)

# ---------------------------------------------------------
# Problem Setup (Problem 1 from Borges & Daripa)
# ---------------------------------------------------------
R = 1.0
RAD_UNIF = 1

def u_true(x, y):
    return 3 * np.exp(x + y) * (x - x**2) * (y - y**2) + 5

def f_rhs(x, y):
    return 6 * np.exp(x + y) * x * y * (-3 + x + y + x * y)

def g_dirichlet(x, y):
    return u_true(x, y)

def u_x(x, y):
    e = np.exp(x + y)
    A = (x - x**2) * (y - y**2)
    A_x = (1 - 2*x) * (y - y**2)
    return 3 * e * (A + A_x)

def u_y(x, y):
    e = np.exp(x + y)
    A = (x - x**2) * (y - y**2)
    A_y = (x - x**2) * (1 - 2*y)
    return 3 * e * (A + A_y)

def g_neumann(x, y):
    return u_x(x, y) * x + u_y(x, y) * y

# ---------------------------------------------------------
# Grid Generation Cache
# ---------------------------------------------------------
ANGLE_CACHE = {}

def get_angle_mesh(method, N, M):
    azu_unif = method["azu_unif"]
    kind = method.get("mesh_kind", "uniform")
    name = method["name"]

    if kind == "uniform":
        return generate_uniform_azimuthal(N)

    if azu_unif == 1:
        key = (name, N)
        if key not in ANGLE_CACHE:
            ANGLE_CACHE[key] = generate_fixed_nonuniform_azimuthal(N, kind=kind)
        return ANGLE_CACHE[key]
    elif azu_unif == 0:
        key = (name, N, M)
        if key not in ANGLE_CACHE:
            ANGLE_CACHE[key] = generate_nonuniform_azimuthal(N, M, kind=kind)
        return ANGLE_CACHE[key]

# ---------------------------------------------------------
# Core Single Test Execution
# ---------------------------------------------------------
def run_case(N, M, method, bc_choice=1, quad_rule=1):
    iRadius = generate_uniform_radial(M, R)
    iAngle = get_angle_mesh(method, N, M)

    x_coord, y_coord = generate_cartesian_grid_on_disk(iAngle, iRadius)
    f_values = generate_grid_values(f_rhs, x_coord, y_coord)
    u_t = generate_grid_values(u_true, x_coord, y_coord)
    
    if bc_choice == 1:
        g_values = generate_grid_values(g_dirichlet, x_coord[:, M-1], y_coord[:, M-1])
    else:
        g_values = generate_grid_values(g_neumann, x_coord[:, M-1], y_coord[:, M-1])

    nudft_flag = method.get("use_nudft", False)
    solver_azu = method.get("solver_azu_unif", method["azu_unif"])
    
    # Calculate zero-mode for Neumann uniqueness
    if bc_choice == 2:
        u_fourier_0 = u_t.mean(axis=0)
    else:
        u_fourier_0 = np.array([])

    t0 = time.perf_counter()
    try:
        u_approx = poisson_solver(
            f_values, g_values, u_fourier_0,
            N, M, iRadius, iAngle, R,
            quad_rule=quad_rule, BC_choice=bc_choice,
            rad_unif=RAD_UNIF,
            azu_unif=solver_azu,
            use_nudft_angular=nudft_flag,
            maxiter_nufft=50, tol_nufft=1e-8
        )
        runtime = time.perf_counter() - t0

        # We only require L_2 relative error metric
        _, _, _, l2_rel = compute_error_metrics(u_approx, u_t, iRadius, iAngle)
    except Exception as exc:
        print(f"  !! ERROR [{method['name']}] N={N} M={M}: {exc}")
        l2_rel = runtime = np.nan

    return dict(
        name=method["name"], label=method["label"],
        N=N, M=M, bc=bc_choice, quad=quad_rule,
        L2_rel=l2_rel, runtime=runtime
    )

# ---------------------------------------------------------
# Table Generation Pipelines
# ---------------------------------------------------------
def run_tests_pipeline(N_values, M_values, fixed_other, methods, test_type="P1_Table1"):
    results = []
    for method in methods:
        print(f"\n{method['label']}")
        if test_type == "P1_Table1":
            for N in N_values:
                for M in M_values:
                    res = run_case(N, M, method, bc_choice=1, quad_rule=1)
                    results.append(res)
                    print(f"  N={N:4d}, M={M:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "P1_Table2":
            for M in M_values:
                for quad in [1, 2]:
                    for bc in [1, 2]:
                        res = run_case(fixed_other, M, method, bc_choice=bc, quad_rule=quad)
                        results.append(res)
                        q_str = "Trapezoidal" if quad == 1 else "Simpson"
                        bc_str = "Dirichlet" if bc == 1 else "Neumann"
                        print(f"  M={M:4d}, {q_str}, {bc_str} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "Accuracy_VaryN":
            for N in N_values:
                res = run_case(N, fixed_other, method, bc_choice=1, quad_rule=1)
                results.append(res)
                print(f"  N={N:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "Accuracy_VaryM":
            for M in M_values:
                res = run_case(fixed_other, M, method, bc_choice=1, quad_rule=1)
                results.append(res)
                print(f"  M={M:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
    return pd.DataFrame(results)

# ---------------------------------------------------------
# Presentation & Rendering
# ---------------------------------------------------------
def render_pivot(df, index_col, columns_col, value_col, title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_col)
    def fmt(x): return "—" if pd.isna(x) else f"{x:.2e}" if "L2" in value_col else f"{x:.3f}"
    display(HTML(pivot.map(fmt).to_html()))

def render_table2(df, title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    df_copy = df.copy()
    df_copy['quad_str'] = df_copy['quad'].map({1: 'Trapezoidal', 2: 'Simpson'})
    df_copy['bc_str'] = df_copy['bc'].map({1: 'Dirichlet', 2: 'Neumann'})
    
    pivot_err = df_copy.pivot_table(index="M", columns=["quad_str", "bc_str"], values="L2_rel")
    pivot_time = df_copy.pivot_table(index="M", columns=["quad_str", "bc_str"], values="runtime")
    
    def fmt(x): return "—" if pd.isna(x) else f"{x:.2e}"
    def fmt_t(x): return "—" if pd.isna(x) else f"{x:.3f}"
    
    print("--- L2 Relative Error ---")
    display(HTML(pivot_err.map(fmt).to_html()))
    print("--- Runtime (s) ---")
    display(HTML(pivot_time.map(fmt_t).to_html()))