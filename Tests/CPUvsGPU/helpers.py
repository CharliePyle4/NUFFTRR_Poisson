import os
import sys
from IPython.display import display, HTML
import numpy as np
import cupy as cp
import pandas as pd
import time
import warnings

from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_cartesian_grid_on_disk,
    generate_grid_values,
    compute_zero_mode
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver


# ---------------------------------------------------------
# Problem Setup (from CPU tests)
# ---------------------------------------------------------
R = 1.0

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
# Core Test Execution
# ---------------------------------------------------------
def run_comparison_case(N, M, bc_choice=1, quad_rule=1, mute=False):
    # --- 1. Setup grid and data on CPU (NumPy) ---
    azu_unif = 2 
    rad_unif = 1
    
    iRadius_np = generate_uniform_radial(M, R)
    iAngle_np = generate_uniform_azimuthal(N)

    x_coord_np, y_coord_np = generate_cartesian_grid_on_disk(iAngle_np, iRadius_np)
    f_values_np = generate_grid_values(f_rhs, x_coord_np, y_coord_np)
    u_t_np = generate_grid_values(u_true, x_coord_np, y_coord_np)
    
    if bc_choice == 1:
        g_values_np = generate_grid_values(g_dirichlet, x_coord_np[:, M-1], y_coord_np[:, M-1])
    else:
        g_values_np = generate_grid_values(g_neumann, x_coord_np[:, M-1], y_coord_np[:, M-1])

    if bc_choice == 2:
        u_fourier_0_arr_np = compute_zero_mode(u_t_np, iAngle_np, azu_unif)
        u_fourier_0_np = u_fourier_0_arr_np[-1]
    else:
        u_fourier_0_np = np.array([])

    # --- 2. Run CPU Solver ---
    t0_cpu = time.perf_counter()
    try:
        u_approx_cpu = poisson_solver(
            f_values_np, g_values_np, u_fourier_0_np,
            N, M, iRadius_np, iAngle_np, R,
            quad_rule=quad_rule, BC_choice=bc_choice,
            rad_unif=rad_unif, azu_unif=azu_unif,
            use_gpu=False
        )
        runtime_cpu = time.perf_counter() - t0_cpu
        _, _, _, l2_rel_cpu = compute_error_metrics(u_approx_cpu, u_t_np, iRadius_np, iAngle_np)
    except Exception as e:
        if not mute:
            print(f"  !! CPU ERROR N={N} M={M}: {e}")
        runtime_cpu, l2_rel_cpu = np.nan, np.nan

    # --- 3. Run GPU Solver ---
    f_values_gpu, g_values_gpu = cp.asarray(f_values_np), cp.asarray(g_values_np)
    u_fourier_0_gpu = cp.asarray(u_fourier_0_np)
    iRadius_gpu, iAngle_gpu = cp.asarray(iRadius_np), cp.asarray(iAngle_np)
    
    cp.cuda.runtime.deviceSynchronize()
    t0_gpu = time.perf_counter()
    try:
        u_approx_gpu_device = poisson_solver(
            f_values_gpu, g_values_gpu, u_fourier_0_gpu,
            N, M, iRadius_gpu, iAngle_gpu, R,
            quad_rule=quad_rule, BC_choice=bc_choice,
            rad_unif=rad_unif, azu_unif=azu_unif,
            use_gpu=True
        )
        cp.cuda.runtime.deviceSynchronize()
        runtime_gpu = time.perf_counter() - t0_gpu
        
        u_approx_gpu_host = cp.asnumpy(u_approx_gpu_device)
        _, _, _, l2_rel_gpu = compute_error_metrics(u_approx_gpu_host, u_t_np, iRadius_np, iAngle_np)

    except Exception as e:
        if not mute:
            print(f"  !! GPU ERROR N={N} M={M}: {e}")
        runtime_gpu, l2_rel_gpu = np.nan, np.nan

    # --- 4. Return results ---
    runtime_diff = runtime_cpu - runtime_gpu if np.isfinite(runtime_cpu) and np.isfinite(runtime_gpu) else np.nan
    return dict(
        N=N, M=M, bc=bc_choice, quad=quad_rule,
        L2_rel_cpu=l2_rel_cpu, runtime_cpu=runtime_cpu,
        L2_rel_gpu=l2_rel_gpu, runtime_gpu=runtime_gpu,
        accuracy_diff=abs(l2_rel_cpu - l2_rel_gpu) if np.isfinite(l2_rel_cpu) and np.isfinite(l2_rel_gpu) else np.nan,
        speedup=runtime_cpu / runtime_gpu if runtime_gpu > 0 and np.isfinite(runtime_cpu) and np.isfinite(runtime_gpu) else np.nan,
        runtime_diff=runtime_diff
    )

# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------
def run_comparison_pipeline(N_values, M_values, test_type="VaryingNM", fixed_val=None, mute=False):
    results = []
    if test_type == "VaryingNM":
        for N in N_values:
            for M in M_values:
                res = run_comparison_case(N, M, mute=mute)
                results.append(res)
                if not mute:
                    print(f"  N={N:4d}, M={M:4d} | Speedup={res['speedup']:.2f}x | Acc. Diff={res['accuracy_diff']:.3e}")
    elif test_type == "VaryingM_BC_Quad":
        N = fixed_val
        for M in M_values:
            for quad in [1, 2]:
                for bc in [1, 2]:
                    res = run_comparison_case(N, M, bc_choice=bc, quad_rule=quad, mute=mute)
                    results.append(res)
                    q_str = "Trap" if quad == 1 else "Simp"
                    bc_str = "Diri" if bc == 1 else "Neum"
                    if not mute:
                        print(f"  N={N}, M={M:4d}, {q_str}, {bc_str} | Speedup={res['speedup']:.2f}x | Acc. Diff={res['accuracy_diff']:.3e}")
    return pd.DataFrame(results)

# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------
def _render_pivot(df, index_col, columns_col, value_cols, title, float_format, col_rename_map=None):
    """Helper to render a single pivot table, potentially with multiple value columns."""
    if title:
        print(f"\n{title}")
    
    # Ensure value_cols is a list for consistency
    if not isinstance(value_cols, list):
        value_cols = [value_cols]

    pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_cols)

    if len(value_cols) > 1:
        # For multi-value tables, create a clean side-by-side view.
        pivot.columns = pivot.columns.swaplevel(0, 1)
        pivot.sort_index(axis=1, level=0, inplace=True)
        
        if col_rename_map:
            # Create a list of new names for the inner level
            new_level_values = [col_rename_map.get(col, col) for col in pivot.columns.get_level_values(1)]
            # Recreate the full MultiIndex with the new names
            pivot.columns = pd.MultiIndex.from_arrays([pivot.columns.get_level_values(0), new_level_values], names=pivot.columns.names)

    display(HTML(pivot.to_html(header=True, float_format=float_format)))

def render_accuracy_tables(df, index_col, columns_col, title_suffix=""):
    """Renders the set of accuracy-related comparison tables."""
    _render_pivot(df, index_col, columns_col, ["L2_rel_cpu", "L2_rel_gpu"], 
                  f"--- Accuracy (L2 Rel. Error){title_suffix} ---",
                  lambda x: f"{x:.2e}", 
                  {'L2_rel_cpu': 'CPU', 'L2_rel_gpu': 'GPU'})
    
    _render_pivot(df, index_col, columns_col, "accuracy_diff", 
                  f"--- Accuracy Difference (Abs){title_suffix} ---",
                  lambda x: f"{x:.2e}")

def render_performance_tables(df, index_col, columns_col, title_suffix=""):
    """Renders the set of performance-related comparison tables."""
    _render_pivot(df, index_col, columns_col, ["runtime_cpu", "runtime_gpu"],
                  f"--- Runtimes (s){title_suffix} ---",
                  lambda x: f"{x:.4f}",
                  {'runtime_cpu': 'CPU', 'runtime_gpu': 'GPU'})

    _render_pivot(df, index_col, columns_col, "runtime_diff",
                  f"--- Runtime Difference (CPU - GPU, s){title_suffix} ---",
                  lambda x: f"{x:.4f}")

    _render_pivot(df, index_col, columns_col, "speedup",
                  f"--- Speedup (CPU / GPU){title_suffix} ---",
                  lambda x: f"{x:.2f}x")

def _prepare_table2_df(df):
    df = df.copy()
    df['quad_str'] = df['quad'].map({1: 'Trapezoidal', 2: 'Simpson'})
    df['bc_str'] = df['bc'].map({1: 'Dirichlet', 2: 'Neumann'})
    return df