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
def run_comparison_case(N, M, bc_choice=1, quad_rule=1, rad_unif=1, mute=False):
    # Contents from helpers.py's run_comparison_case
    azu_unif = 2 
    if rad_unif == 1:
        iRadius_np = generate_uniform_radial(M, R)
    else:
        # Default nonuniform grid is 'sqrt' mapping
        iRadius_np = generate_nonuniform_radial(M, R, mapping='sqrt')

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
    t0_cpu = time.perf_counter()
    try:
        u_approx_cpu = poisson_solver(f_values_np, g_values_np, u_fourier_0_np, N, M, iRadius_np, iAngle_np, R, quad_rule=quad_rule, BC_choice=bc_choice, rad_unif=rad_unif, azu_unif=azu_unif, use_gpu=False)
        runtime_cpu = time.perf_counter() - t0_cpu
        _, _, _, l2_rel_cpu = compute_error_metrics(u_approx_cpu, u_t_np, iRadius_np, iAngle_np)
    except Exception as e:
        if not mute: print(f"  !! CPU ERROR N={N} M={M}: {e}")
        runtime_cpu, l2_rel_cpu = np.nan, np.nan
    f_values_gpu, g_values_gpu = cp.asarray(f_values_np), cp.asarray(g_values_np)
    u_fourier_0_gpu = cp.asarray(u_fourier_0_np)
    iRadius_gpu, iAngle_gpu = cp.asarray(iRadius_np), cp.asarray(iAngle_np)
    cp.cuda.runtime.deviceSynchronize()
    t0_gpu = time.perf_counter()
    try:
        u_approx_gpu_device = poisson_solver(f_values_gpu, g_values_gpu, u_fourier_0_gpu, N, M, iRadius_gpu, iAngle_gpu, R, quad_rule=quad_rule, BC_choice=bc_choice, rad_unif=rad_unif, azu_unif=azu_unif, use_gpu=True)
        cp.cuda.runtime.deviceSynchronize()
        runtime_gpu = time.perf_counter() - t0_gpu
        u_approx_gpu_host = cp.asnumpy(u_approx_gpu_device)
        _, _, _, l2_rel_gpu = compute_error_metrics(u_approx_gpu_host, u_t_np, iRadius_np, iAngle_np)
    except Exception as e:
        if not mute: print(f"  !! GPU ERROR N={N} M={M}: {e}")
        runtime_gpu, l2_rel_gpu = np.nan, np.nan
    runtime_diff = runtime_cpu - runtime_gpu if np.isfinite(runtime_cpu) and np.isfinite(runtime_gpu) else np.nan
    return dict(N=N, M=M, bc=bc_choice, quad=quad_rule, rad_unif=rad_unif, L2_rel_cpu=l2_rel_cpu, runtime_cpu=runtime_cpu, L2_rel_gpu=l2_rel_gpu, runtime_gpu=runtime_gpu, accuracy_diff=abs(l2_rel_cpu - l2_rel_gpu) if np.isfinite(l2_rel_cpu) and np.isfinite(l2_rel_gpu) else np.nan, speedup=runtime_cpu / runtime_gpu if runtime_gpu > 0 and np.isfinite(runtime_cpu) and np.isfinite(runtime_gpu) else np.nan, runtime_diff=runtime_diff)

# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------
def run_comparison_pipeline(N_values, M_values, test_type="VaryingNM", fixed_val=None, mute=False):
    # Contents from helpers.py's run_comparison_pipeline
    results = []
    if test_type == "VaryingNM":
        for N in N_values:
            for M in M_values:
                res = run_comparison_case(N, M, rad_unif=1, mute=mute)
                results.append(res)
                if not mute: print(f"  N={N:4d}, M={M:4d} | Speedup={res['speedup']:.2f}x | Acc. Diff={res['accuracy_diff']:.3e}")
    elif test_type == "VaryingM_BC_Quad":
        N = fixed_val
        for M in M_values:
            for quad in [1, 2]:
                for bc in [1, 2]:
                    res = run_comparison_case(N, M, bc_choice=bc, quad_rule=quad, rad_unif=1, mute=mute)
                    results.append(res)
                    q_str = "Trap" if quad == 1 else "Simp"
                    bc_str = "Diri" if bc == 1 else "Neum"
                    if not mute: print(f"  N={N}, M={M:4d}, {q_str}, {bc_str} | Speedup={res['speedup']:.2f}x | Acc. Diff={res['accuracy_diff']:.3e}")
    elif test_type == "VaryingM_BC_Quad_Nonunif":
        N = fixed_val
        for M in M_values:
            for quad in [1, 2]:
                for bc in [1, 2]:
                    res = run_comparison_case(N, M, bc_choice=bc, quad_rule=quad, rad_unif=0, mute=mute)
                    results.append(res)
                    q_str = "Trap" if quad == 1 else "Simp"
                    bc_str = "Diri" if bc == 1 else "Neum"
                    if not mute: print(f"  N={N}, M={M:4d}, {q_str}, {bc_str}, Nonunif | Speedup={res['speedup']:.2f}x | Acc. Diff={res['accuracy_diff']:.3e}")
    return pd.DataFrame(results)

# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------
def _render_pivot(df, index_col, columns_col, value_cols, title, float_format, col_rename_map=None):
    if title: print(f"\n{title}")
    if not isinstance(value_cols, list): value_cols = [value_cols]
    pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_cols)
    if len(value_cols) > 1 and pivot.columns.nlevels > 1:
        new_order = list(range(1, pivot.columns.nlevels)) + [0]
        pivot.columns = pivot.columns.reorder_levels(new_order)
        pivot.sort_index(axis=1, inplace=True)
        if col_rename_map:
            last_level_idx = pivot.columns.nlevels - 1
            new_level_values = [col_rename_map.get(item, item) for item in pivot.columns.levels[last_level_idx]]
            pivot.columns = pivot.columns.set_levels(new_level_values, level=last_level_idx)
    display(HTML(pivot.to_html(header=True, float_format=float_format)))

def render_accuracy_tables(df, index_col, columns_col, title_suffix=""):
    _render_pivot(df, index_col, columns_col, ["L2_rel_cpu", "L2_rel_gpu"], f"--- Accuracy (L2 Rel. Error){title_suffix} ---", lambda x: f"{x:.2e}", {'L2_rel_cpu': 'CPU', 'L2_rel_gpu': 'GPU'})
    _render_pivot(df, index_col, columns_col, "accuracy_diff", f"--- Accuracy Difference (Abs){title_suffix} ---", lambda x: f"{x:.2e}")

def render_performance_tables(df, index_col, columns_col, title_suffix=""):
    _render_pivot(df, index_col, columns_col, ["runtime_cpu", "runtime_gpu"], f"--- Runtimes (s){title_suffix} ---", lambda x: f"{x:.4f}", {'runtime_cpu': 'CPU', 'runtime_gpu': 'GPU'})
    _render_pivot(df, index_col, columns_col, "runtime_diff", f"--- Runtime Difference (CPU - GPU, s){title_suffix} ---", lambda x: f"{x:.4f}")
    _render_pivot(df, index_col, columns_col, "speedup", f"--- Speedup (CPU / GPU){title_suffix} ---", lambda x: f"{x:.2f}x")

def _prepare_table2_df(df):
    df = df.copy()
    df['quad_str'] = df['quad'].map({1: 'Trapezoidal', 2: 'Simpson'})
    df['bc_str'] = df['bc'].map({1: 'Dirichlet', 2: 'Neumann'})
    if 'rad_unif' in df.columns:
        df['grid_str'] = df['rad_unif'].map({1: 'Uniform', 0: 'Nonuniform'})
    return df

# ---------------------------------------------------------
# Plot Helpers
# ---------------------------------------------------------
import matplotlib.pyplot as plt

def _safe_plot_series(ax, x, y, label, marker="o", linestyle="-"):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if np.any(mask):
        ax.plot(x[mask], y[mask], marker=marker, linestyle=linestyle, label=label)

def _style_runtime_axis(ax, title, xlabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runtime (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_runtime_vs_n(df_nm):
    df_plot = df_nm.copy()

    m_min = df_plot["M"].min()
    m_max = df_plot["M"].max()

    df_m_min = df_plot[df_plot["M"] == m_min].sort_values("N")
    df_m_max = df_plot[df_plot["M"] == m_max].sort_values("N")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    _safe_plot_series(axes[0], df_m_min["N"], df_m_min["runtime_cpu"], label="CPU", marker="o")
    _safe_plot_series(axes[0], df_m_min["N"], df_m_min["runtime_gpu"], label="GPU", marker="s")
    _style_runtime_axis(axes[0], f"CPU vs GPU Runtime for Varying N (M={m_min})", "N")

    _safe_plot_series(axes[1], df_m_max["N"], df_m_max["runtime_cpu"], label="CPU", marker="o")
    _safe_plot_series(axes[1], df_m_max["N"], df_m_max["runtime_gpu"], label="GPU", marker="s")
    _style_runtime_axis(axes[1], f"CPU vs GPU Runtime for Varying N (M={m_max})", "N")

    plt.tight_layout()
    plt.show()

def plot_runtime_vs_m(df_nm):
    df_plot = df_nm.copy()

    n_min = df_plot["N"].min()
    n_max = df_plot["N"].max()

    df_n_min = df_plot[df_plot["N"] == n_min].sort_values("M")
    df_n_max = df_plot[df_plot["N"] == n_max].sort_values("M")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    _safe_plot_series(axes[0], df_n_min["M"], df_n_min["runtime_cpu"], label="CPU", marker="o")
    _safe_plot_series(axes[0], df_n_min["M"], df_n_min["runtime_gpu"], label="GPU", marker="s")
    _style_runtime_axis(axes[0], f"CPU vs GPU Runtime for Varying M (N={n_min})", "M")

    _safe_plot_series(axes[1], df_n_max["M"], df_n_max["runtime_cpu"], label="CPU", marker="o")
    _safe_plot_series(axes[1], df_n_max["M"], df_n_max["runtime_gpu"], label="GPU", marker="s")
    _style_runtime_axis(axes[1], f"CPU vs GPU Runtime for Varying M (N={n_max})", "M")

    plt.tight_layout()
    plt.show()

def plot_runtime_conditions_vs_m(df_bc_quad):
    df_plot = _prepare_table2_df(df_bc_quad).copy()
    if 'grid_str' in df_plot.columns:
        df_plot["condition"] = df_plot["grid_str"] + " + " + df_plot["quad_str"] + " + " + df_plot["bc_str"]
    else:
        df_plot["condition"] = df_plot["quad_str"] + " + " + df_plot["bc_str"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for condition, subdf in df_plot.groupby("condition"):
        subdf = subdf.sort_values("M")
        _safe_plot_series(axes[0], subdf["M"], subdf["runtime_cpu"], label=condition, marker="o")
        _safe_plot_series(axes[1], subdf["M"], subdf["runtime_gpu"], label=condition, marker="s")

    _style_runtime_axis(axes[0], "CPU Runtime by Condition for Varying M", "M")
    _style_runtime_axis(axes[1], "GPU Runtime by Condition for Varying M", "M")

    plt.tight_layout()
    plt.show()