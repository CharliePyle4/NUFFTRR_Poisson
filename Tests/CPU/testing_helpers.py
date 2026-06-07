import os
import sys
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import sympy as sp
from IPython.display import display
import matplotlib.pyplot as plt
import time
import warnings

from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_nonuniform_radial,
    generate_uniform_azimuthal,
    generate_fixed_nonuniform_azimuthal,
    generate_nonuniform_azimuthal,
    generate_cartesian_grid_on_disk,
    generate_grid_values,
    compute_zero_mode
)
from Poisson_Solver.visualization import compute_error_metrics, plot_on_disk_with_error
from Poisson_Solver.poisson_solver import poisson_solver


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
def run_case(N, M, method, bc_choice=1, quad_rule=1, mute=False):
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
        u_fourier_0_arr = compute_zero_mode(u_t, iAngle, method["azu_unif"])
        u_fourier_0 = u_fourier_0_arr[-1]  # Pass u_0(R) to fix the constant
    else:
        u_fourier_0 = np.array([])

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        if mute:
            warnings.simplefilter("ignore")

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
            if not mute:
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
def run_tests_pipeline(N_values, M_values, fixed_other, methods, test_type="P1_Table1", mute=False):
    results = []
    for method in methods:
        if not mute:
            print(f"\n{method['label']}")
        if test_type == "P1_Table1":
            for N in N_values:
                for M in M_values:
                    res = run_case(N, M, method, bc_choice=1, quad_rule=1, mute=mute)
                    results.append(res)
                    if not mute:
                        print(f"  N={N:4d}, M={M:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "P1_Table2":
            for M in M_values:
                for quad in [1, 2]:
                    for bc in [1, 2]:
                        res = run_case(fixed_other, M, method, bc_choice=bc, quad_rule=quad, mute=mute)
                        results.append(res)
                        q_str = "Trapezoidal" if quad == 1 else "Simpson"
                        bc_str = "Dirichlet" if bc == 1 else "Neumann"
                        if not mute:
                            print(f"  M={M:4d}, {q_str}, {bc_str} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "Accuracy_VaryN":
            for N in N_values:
                res = run_case(N, fixed_other, method, bc_choice=1, quad_rule=1, mute=mute)
                results.append(res)
                if not mute:
                    print(f"  N={N:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "Accuracy_VaryM":
            for M in M_values:
                res = run_case(fixed_other, M, method, bc_choice=1, quad_rule=1, mute=mute)
                results.append(res)
                if not mute:
                    print(f"  M={M:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
    return pd.DataFrame(results)

# ---------------------------------------------------------
# Presentation & Rendering
# ---------------------------------------------------------
def render_pivot(df, index_col, columns_col, value_col, title):
    """Base pivoting and formatting utility."""
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    
    # Create the pivot table
    pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_col, sort=True)
    
    # Sort columns to ensure Uniform -> NUDFT -> NUFFT order if labels exist
    cols = pivot.columns.tolist()
    preferred = ["Uniform", "/ FFT", "/ NUDFT", "/ NUFFT"]
    sorted_cols = []
    
    # This handles both single strings and tuples (for MultiIndex columns like [label, M])
    for p in preferred:
        for c in cols:
            c_str = str(c)
            if p in c_str and c not in sorted_cols:
                sorted_cols.append(c)
    
    # Special handling for Table 2 column grouping (Trapezoidal/Simpson/Dirichlet/Neumann)
    # If we didn't find preferred strings, use original order
    if not sorted_cols:
        sorted_cols = cols

    for c in cols:
        if c not in sorted_cols:
            sorted_cols.append(c)
            
    pivot = pivot[sorted_cols]

    def fmt(x):
        if pd.isna(x): return "—"
        return f"{x:.2e}" if ("L2" in value_col or (isinstance(x, (float, np.float64)) and x < 1e-3)) else f"{x:.4f}"
        
    display(HTML(pivot.map(fmt).to_html()))


def render_accuracy(df, index_col, columns_col, title_prefix):
    """Renders condensed tables for L2 relative error."""
    render_pivot(df, index_col, columns_col, "L2_rel", f"{title_prefix} Accuracy")

def render_runtime(df, index_col, columns_col, title_prefix):
    """Renders condensed tables for runtime."""
    render_pivot(df, index_col, columns_col, "runtime", f"{title_prefix} Runtime (s)")

def _prepare_table2_df(df):
    """Helper to add string labels for Table 2 parameters."""
    df = df.copy()
    df['quad_str'] = df['quad'].map({1: 'Trapezoidal', 2: 'Simpson'})
    df['bc_str'] = df['bc'].map({1: 'Dirichlet', 2: 'Neumann'})
    return df

def render_table2_accuracy(df, title_prefix="Table 2"):
    """Renders Problem 1 Table 2 accuracy with methods grouped."""
    df_fmt = _prepare_table2_df(df)
    render_accuracy(df_fmt, "M", ["label", "quad_str", "bc_str"], title_prefix)

def render_table2_runtime(df, title_prefix="Table 2"):
    """Renders Problem 1 Table 2 runtime with methods grouped."""
    df_fmt = _prepare_table2_df(df)
    render_runtime(df_fmt, "M", ["label", "quad_str", "bc_str"], title_prefix)

# ---------------------------------------------------------
# New Visualization & Plotting Helpers
# ---------------------------------------------------------

def run_and_plot_case(N, M, method, bc_choice=1, quad_rule=1):
    """
    Runs a single test case and plots the true solution, approximate solution,
    and the pointwise error.
    """
    print(f"Running case for N={N}, M={M}, method='{method['name']}', BC={'Dirichlet' if bc_choice==1 else 'Neumann'}")

    # --- Setup grid and problem values (mirrors run_case) ---
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
    
    if bc_choice == 2:
        u_fourier_0_arr = compute_zero_mode(u_t, iAngle, method["azu_unif"])
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    # --- Run solver ---
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
        print(f"Solver runtime: {runtime:.4f}s")

        # --- Compute, print, and plot errors ---
        _, _, _, l2_rel = compute_error_metrics(u_approx, u_t, iRadius, iAngle)
        print(f"L2 Relative Error: {l2_rel:.4e}")

        plot_on_disk_with_error(x_coord, y_coord, u_approx, u_t)

    except Exception as exc:
        print(f"  !! ERROR [{method['name']}] N={N} M={M}: {exc}")


def plot_convergence(df, x_axis='N', title=None, log_y=True, log_x=False):
    """
    Plots convergence curves from a results DataFrame generated by run_tests_pipeline.
    """
    if df.empty:
        print("DataFrame is empty, nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    for name, group in df.groupby('name'):
        group = group.sort_values(by=x_axis)
        label = group['label'].iloc[0]
        plt.plot(group[x_axis], group['L2_rel'], 'o-', label=label)

    plt.xlabel(x_axis)
    plt.ylabel('L2 Relative Error')
    
    if log_y: plt.yscale('log')
    if log_x: plt.xscale('log')
        
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.title(title if title else f'Convergence vs. {x_axis}')
    plt.show()