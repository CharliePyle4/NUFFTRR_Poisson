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
from Poisson_Solver.visualization import compute_error_metrics
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

def run_and_plot_errors_vary_m(N_fixed, M_values, methods, bc_choice=1, quad_rule=1):
    """
    Runs test cases for a fixed N and varying M for different methods,
    and plots the pointwise error for each in a grid structure.
    """
    num_methods = len(methods)
    num_m = len(M_values)
    
    fig, axes = plt.subplots(
        num_methods, 
        num_m, 
        figsize=(5 * num_m, 4.5 * num_methods), 
        subplot_kw={'projection': '3d'}
    )
    
    # Ensure axes is always a 2D array for consistent indexing
    if num_methods == 1 and num_m == 1:
        axes = np.array([[axes]])
    elif num_methods == 1:
        axes = np.array([axes])
    elif num_m == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"Pointwise Error for N={N_fixed} (BC={'Dirichlet' if bc_choice==1 else 'Neumann'}, Quad={'Trap.' if quad_rule==1 else 'Simp.'})", fontsize=16)

    for i, method in enumerate(methods):
        for j, M in enumerate(M_values):
            ax = axes[i, j]
            
            # --- Run solver logic ---
            iRadius = generate_uniform_radial(M, R)
            iAngle = get_angle_mesh(method, N_fixed, M)

            x_coord, y_coord = generate_cartesian_grid_on_disk(iAngle, iRadius)
            f_values = generate_grid_values(f_rhs, x_coord, y_coord)
            u_t = generate_grid_values(u_true, x_coord, y_coord)
            
            g_func = g_dirichlet if bc_choice == 1 else g_neumann
            g_values = generate_grid_values(g_func, x_coord[:, -1], y_coord[:, -1])

            u_fourier_0 = compute_zero_mode(u_t, iAngle, method["azu_unif"])[-1] if bc_choice == 2 else np.array([])

            try:
                u_approx = poisson_solver(
                    f_values, g_values, u_fourier_0, N_fixed, M, iRadius, iAngle, R,
                    quad_rule=quad_rule, BC_choice=bc_choice, rad_unif=RAD_UNIF, azu_unif=method.get("solver_azu_unif", method["azu_unif"]),
                    use_nudft_angular=method.get("use_nudft", False)
                )
                _, _, _, l2_rel = compute_error_metrics(u_approx, u_t, iRadius, iAngle)
                
                ptwise_error = np.abs(u_t - u_approx)
                Xp, Yp = np.vstack([x_coord, x_coord[0, :]]), np.vstack([y_coord, y_coord[0, :]])
                Ep = np.vstack([ptwise_error, ptwise_error[0, :]])

                ax.plot_surface(Xp, Yp, Ep, cmap="viridis")
                ax.set_title(f"{method['label']}\nM={M}, L2_rel={l2_rel:.2e}")

            except Exception as exc:
                print(f"  !! ERROR [{method['name']}] N={N_fixed} M={M}: {exc}")
                ax.text(0.5, 0.5, 0.5, "ERROR", transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f"{method['label']}\nM={M}")
                ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


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





    # ---------------------------------------------------------
# Radial Grid Configs
# (add alongside your existing METHODS list)
# ---------------------------------------------------------

def make_radial_method(name, label, rad_unif, rad_mapping=None, azu_unif=2):
    """
    Build a method dict for a radial grid variant.
    Compatible with run_case() — uses uniform azimuthal (azu_unif=2).
    rad_mapping is passed to generate_nonuniform_radial when rad_unif=0.
    """
    return dict(
        name=name,
        label=label,
        azu_unif=azu_unif,
        mesh_kind="uniform",          # angular stays uniform
        rad_unif=rad_unif,
        rad_mapping=rad_mapping,      # extra key, read by run_case_radial below
    )

RADIAL_METHODS = [
    make_radial_method("uniform",  "Uniform Radial",      rad_unif=1, rad_mapping=None),
    make_radial_method("sqrt",     "Nonuniform (sqrt)",   rad_unif=0, rad_mapping="sqrt"),
    make_radial_method("random",  "Nonuniform (rand)",   rad_unif=0, rad_mapping="random")
]


# ---------------------------------------------------------
# Core runner — radial grid variant
# Mirrors run_case() but respects rad_unif / rad_mapping.
# ---------------------------------------------------------

def run_case_radial(N, M, method, bc_choice=1, quad_rule=1, mute=False):
    """
    Like run_case(), but generates the radial grid from method['rad_mapping']
    (nonuniform) or linspace (uniform) based on method['rad_unif'].
    """
    rad_unif = method["rad_unif"]
    rad_mapping = method.get("rad_mapping")

    if rad_unif == 1 or rad_mapping is None:
        iRadius = generate_uniform_radial(M, R)
    else:
        iRadius = generate_nonuniform_radial(M, R, mapping=rad_mapping)

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

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u_approx = poisson_solver(
                f_values, g_values, u_fourier_0,
                N, M, iRadius, iAngle, R,
                quad_rule=quad_rule, BC_choice=bc_choice,
                rad_unif=rad_unif,
                azu_unif=solver_azu,
                use_nudft_angular=nudft_flag,
                maxiter_nufft=50, tol_nufft=1e-8
            )
            runtime = time.perf_counter() - t0
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
# Pipeline: N x M sweep across radial grid types
# ---------------------------------------------------------

def run_NM_radial_sweep(N_values, M_values, methods=None,
                         bc_choice=1, quad_rule=1, mute=False):
    """
    Sweeps all (N, M) combos for each radial method.
    Returns a DataFrame with columns: name, label, N, M, bc, quad, L2_rel, runtime.
    """
    if methods is None:
        methods = RADIAL_METHODS
    rows = []
    for method in methods:
        if not mute:
            print(f"\n── {method['label']} ──")
        for N in N_values:
            for M in M_values:
                r = run_case_radial(N, M, method, bc_choice=bc_choice,
                                    quad_rule=quad_rule, mute=mute)
                rows.append(r)
                if not mute:
                    print(f"  N={N:4d}, M={M:4d} | L2_rel={r['L2_rel']:.3e} | t={r['runtime']:.3f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Pipeline: BC x quad sweep across radial grid types
# ---------------------------------------------------------

def run_bc_quad_radial_sweep(N_fixed, M_values, methods=None, mute=False):
    """
    For a fixed N, sweeps M x {Trap, Simp} x {Dirichlet, Neumann} for each radial method.
    Returns a DataFrame with added quad_str and bc_str columns.
    """
    if methods is None:
        methods = RADIAL_METHODS
    rows = []
    for method in methods:
        if not mute:
            print(f"\n── {method['label']} ──")
        for M in M_values:
            for quad in [1, 2]:
                for bc in [1, 2]:
                    r = run_case_radial(N_fixed, M, method,
                                        bc_choice=bc, quad_rule=quad, mute=mute)
                    rows.append(r)
                    if not mute:
                        q_s = "Trap" if quad == 1 else "Simp"
                        b_s = "Diri" if bc == 1 else "Neum"
                        print(f"  M={M:4d}, {q_s}, {b_s} | L2_rel={r['L2_rel']:.3e} | t={r['runtime']:.3f}s")
    df = pd.DataFrame(rows)
    df["quad_str"] = df["quad"].map({1: "Trapezoidal", 2: "Simpson"})
    df["bc_str"]   = df["bc"].map({1: "Dirichlet",    2: "Neumann"})
    return df


# ---------------------------------------------------------
# Rendering: reuse render_pivot / render_accuracy / render_runtime
# These wrappers just call your existing functions with the right args.
# ---------------------------------------------------------

def render_NM_radial_accuracy(df, title_suffix=""):
    """L2 accuracy table: rows=N, cols=M, one table per radial method."""
    for lbl in df["label"].unique():
        sub = df[df["label"] == lbl]
        render_accuracy(sub, "N", "M", f"{lbl}{title_suffix}")

def render_NM_radial_runtime(df, title_suffix=""):
    """Runtime table: rows=N, cols=M, one table per radial method."""
    for lbl in df["label"].unique():
        sub = df[df["label"] == lbl]
        render_runtime(sub, "N", "M", f"{lbl}{title_suffix}")

def render_bc_quad_radial_accuracy(df, title_suffix=""):
    """Accuracy table: rows=M, cols=[label, quad_str, bc_str]."""
    df_fmt = _prepare_table2_df(df) if "quad_str" not in df.columns else df
    render_accuracy(df_fmt, "M", ["label", "quad_str", "bc_str"],
                    f"BC × Quad Accuracy{title_suffix}")

def render_bc_quad_radial_runtime(df, title_suffix=""):
    """Runtime table: rows=M, cols=[label, quad_str, bc_str]."""
    df_fmt = _prepare_table2_df(df) if "quad_str" not in df.columns else df
    render_runtime(df_fmt, "M", ["label", "quad_str", "bc_str"],
                   f"BC × Quad Runtime{title_suffix}")


# ---------------------------------------------------------
# Plots: uniform vs nonuniform comparison
# ---------------------------------------------------------

def _safe_plot_radial(ax, x, y, label, **kw):
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.any():
        ax.plot(x[mask], y[mask], label=label, **kw)


def plot_radial_accuracy_vs_M(df_nm, N_fixed, methods=None):
    """
    Log-scale L2 accuracy vs M for each radial grid type at a fixed N.
    Uniform is solid; nonuniform are dashed.
    """
    if methods is None:
        methods = RADIAL_METHODS
    sub = df_nm[df_nm["N"] == N_fixed]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        lbl = method["label"]
        d = sub[sub["label"] == lbl].sort_values("M")
        ls = "-" if method["rad_unif"] == 1 else "--"
        _safe_plot_radial(ax, d["M"], d["L2_rel"], label=lbl, marker="o", linestyle=ls)
    ax.set_yscale("log")
    ax.set_xlabel("M  (radial points)")
    ax.set_ylabel("L2 Relative Error")
    ax.set_title(f"Accuracy vs M — Uniform vs Nonuniform Radial  (N = {N_fixed})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_radial_accuracy_vs_N(df_nm, M_fixed, methods=None):
    """
    Log-scale L2 accuracy vs N for each radial grid type at a fixed M.
    """
    if methods is None:
        methods = RADIAL_METHODS
    sub = df_nm[df_nm["M"] == M_fixed]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        lbl = method["label"]
        d = sub[sub["label"] == lbl].sort_values("N")
        ls = "-" if method["rad_unif"] == 1 else "--"
        _safe_plot_radial(ax, d["N"], d["L2_rel"], label=lbl, marker="s", linestyle=ls)
    ax.set_yscale("log")
    ax.set_xlabel("N  (angular points)")
    ax.set_ylabel("L2 Relative Error")
    ax.set_title(f"Accuracy vs N — Uniform vs Nonuniform Radial  (M = {M_fixed})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_radial_runtime_vs_M(df_nm, N_fixed, methods=None):
    """Runtime vs M for each radial grid type at a fixed N."""
    if methods is None:
        methods = RADIAL_METHODS
    sub = df_nm[df_nm["N"] == N_fixed]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        lbl = method["label"]
        d = sub[sub["label"] == lbl].sort_values("M")
        ls = "-" if method["rad_unif"] == 1 else "--"
        _safe_plot_radial(ax, d["M"], d["runtime"], label=lbl, marker="o", linestyle=ls)
    ax.set_xlabel("M  (radial points)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime vs M  (N = {N_fixed})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_bc_quad_radial_accuracy(df_bq, method_label):
    """
    2×2 panel: one subplot per (BC, quad) combo, L2 accuracy vs M,
    for a single named radial grid type.
    """
    sub = df_bq[df_bq["label"] == method_label]
    combos = [(1, 1, "Dirichlet + Trapezoidal"),
              (1, 2, "Dirichlet + Simpson"),
              (2, 1, "Neumann + Trapezoidal"),
              (2, 2, "Neumann + Simpson")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Accuracy vs M  —  {method_label}", fontsize=13)
    for ax, (bc, quad, ttl) in zip(axes.ravel(), combos):
        d = sub[(sub["bc"] == bc) & (sub["quad"] == quad)].sort_values("M")
        _safe_plot_radial(ax, d["M"], d["L2_rel"], label="L2 rel", marker="o")
        ax.set_yscale("log")
        ax.set_title(ttl)
        ax.set_xlabel("M")
        ax.set_ylabel("L2 Relative Error")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bc_quad_comparison_vs_M(df_bq, N_fixed, methods=None):
    """
    4-panel figure comparing all radial grid types.
    One panel per (BC, quad) combo, all methods overlaid, vs M.
    """
    if methods is None:
        methods = RADIAL_METHODS
    combos = [(1, 1, "Dirichlet + Trapezoidal"),
              (1, 2, "Dirichlet + Simpson"),
              (2, 1, "Neumann + Trapezoidal"),
              (2, 2, "Neumann + Simpson")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"All Grid Types — BC × Quad Accuracy  (N = {N_fixed})", fontsize=13)
    for ax, (bc, quad, ttl) in zip(axes.ravel(), combos):
        for method in methods:
            lbl = method["label"]
            d = df_bq[(df_bq["label"] == lbl) &
                      (df_bq["bc"] == bc) &
                      (df_bq["quad"] == quad)].sort_values("M")
            ls = "-" if method["rad_unif"] == 1 else "--"
            _safe_plot_radial(ax, d["M"], d["L2_rel"], label=lbl,
                              marker="o", linestyle=ls)
        ax.set_yscale("log")
        ax.set_title(ttl)
        ax.set_xlabel("M")
        ax.set_ylabel("L2 Relative Error")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()