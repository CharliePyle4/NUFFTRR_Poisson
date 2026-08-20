import os
import sys
from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from IPython.display import display, HTML
import numpy as np
import pandas as pd
import sympy as sp
from IPython.display import display
import matplotlib.pyplot as plt
import time
import warnings
from tqdm.auto import tqdm

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
# Optimal Lambda & Parameter Table Resolution
# ---------------------------------------------------------
def compute_optimal_lambda(theta_j, N, M, is_nudft=False):
    """
    Pre-calculates the optimal lambda / reg_param from grid parameters (N, M, theta).
    Balances radial quadrature discretization error O(M^-2.5) with non-uniform angular
    Beurling-Landau frame distortion rho = max(d_theta) / (2*pi/N).
    """
    if theta_j is None:
        return 1e-12
    theta_arr = np.asarray(theta_j, dtype=float)
    if theta_arr.size == 0:
        return 1e-12
    if theta_arr.ndim > 1:
        theta_arr = theta_arr[:, 0]
    
    theta_sorted = np.sort(theta_arr % (2.0 * np.pi))
    d_theta = np.diff(np.concatenate([theta_sorted, [theta_sorted[0] + 2.0 * np.pi]]))
    avg_gap = 2.0 * np.pi / N
    rho = np.max(d_theta) / avg_gap  # rho >= 1.0 (rho=1.0 is uniform)
    
    if is_nudft:
        if rho <= 1.5:
            return 1e-12
        else:
            cond = 1e-13 * (rho * N) ** 2.0
            return float(np.clip(cond, 1e-12, 1e-6))
    else:
        # NUFFT: balance radial error floor O(M^-2.5) with angular frame distortion
        base_lambda = 0.05 * (M ** -2.5)
        
        if rho > 1.5:
            distortion_factor = ((rho - 1.0) ** 2.0) * (N / 32.0)
            reg = base_lambda * (1.0 + 50.0 * distortion_factor)
            return float(np.clip(reg, 1e-10, 1e-4))
        else:
            return float(np.clip(base_lambda, 1e-11, 1e-6))


def resolve_reg_param(cfg_reg, method, N, M, iAngle=None):
    """
    Resolves the regularization parameter (lambda) for a specific run (N, M, method).
    
    Supports:
    1. Single float / int (e.g. 1e-12)
    2. Dict keyed by method name, label, or mesh kind:
       e.g. {'Fixed-rand-NUFFT': 1e-8, 'sine': 1e-6}
    3. Dict keyed by (method, N), (method, M), (method, N, M), or (N, M):
       e.g. {('Fixed-sine-NUFFT', 128): 1e-4, (32, 32): 1e-10}
    4. Nested dict:
       e.g. {'Fixed-sine-NUFFT': {32: 1e-6, 64: 1e-6, 128: 1e-4}}
    5. List / tuple / 1D array (vector matching standard resolutions [32, 64, 128, 256, 512])
    6. pandas DataFrame (table where index is N or M and columns are methods)
    7. Callable function: fn(N, M, method)
    8. Method-level override: method.get('reg_param')
    """
    # 1. Check method-level override if specified directly in method dictionary
    if "reg_param" in method and method["reg_param"] is not None:
        cfg = method["reg_param"]
    elif "lambda" in method and method["lambda"] is not None:
        cfg = method["lambda"]
    else:
        cfg = cfg_reg

    if cfg is None:
        return 1e-12

    # If callable
    if callable(cfg):
        return float(cfg(N, M, method))

    # If simple float / int
    if isinstance(cfg, (int, float, np.floating, np.integer)):
        return float(cfg)

    m_name = method.get("name", "")
    m_label = method.get("label", "")
    m_kind = method.get("mesh_kind", "")
    is_nudft = method.get("use_nudft", False)

    # If string 'auto' or 'optimal'
    if isinstance(cfg, str):
        if cfg.lower() in ("auto", "optimal"):
            return compute_optimal_lambda(iAngle, N, M, is_nudft=is_nudft)
        try:
            return float(cfg)
        except ValueError:
            return 1e-12

    # If pandas DataFrame
    if isinstance(cfg, pd.DataFrame):
        for col_key in (m_name, m_label, m_kind):
            if col_key in cfg.columns:
                if N in cfg.index:
                    return float(cfg.loc[N, col_key])
                elif M in cfg.index:
                    return float(cfg.loc[M, col_key])
                elif (N, M) in cfg.index:
                    return float(cfg.loc[(N, M), col_key])

    # If dictionary
    if isinstance(cfg, dict):
        # Check specific tuple keys first: (name, N, M), (name, N), (name, M), (N, M)
        for key in [
            (m_name, N, M), (m_label, N, M), (m_kind, N, M),
            (m_name, N), (m_label, N), (m_kind, N),
            (m_name, M), (m_label, M), (m_kind, M),
            (N, M), N, M
        ]:
            if key in cfg:
                val = cfg[key]
                if isinstance(val, (int, float, np.floating, np.integer)):
                    return float(val)
                elif isinstance(val, dict):
                    if N in val: return float(val[N])
                    if M in val: return float(val[M])

        # Check method name / label / kind keys
        for key in (m_name, m_label, m_kind):
            if key in cfg:
                val = cfg[key]
                if isinstance(val, (int, float, np.floating, np.integer)):
                    return float(val)
                elif isinstance(val, dict):
                    # 1. Exact (N, M) pair
                    if (N, M) in val:
                        return float(val[(N, M)])
                    # 2. Both "N" and "M" subdictionaries are provided
                    if "N" in val and "M" in val:
                        if N != 32 and M == 32 and N in val["N"]:
                            return float(val["N"][N])
                        elif M != 32 and N == 32 and M in val["M"]:
                            return float(val["M"][M])
                        elif N in val["N"]:
                            return float(val["N"][N])
                        elif M in val["M"]:
                            return float(val["M"][M])
                    # 3. Explicit "N" subdict
                    elif "N" in val and N in val["N"]:
                        return float(val["N"][N])
                    # 4. Explicit "M" subdict
                    elif "M" in val and M in val["M"]:
                        return float(val["M"][M])
                    # 5. Flat dictionary keyed by integers (e.g. {32: ..., 64: ...})
                    elif N in val:
                        return float(val[N])
                    elif M in val:
                        return float(val[M])
                elif isinstance(val, (list, tuple, np.ndarray)):
                    res_list = [32, 64, 128, 256, 512]
                    if N in res_list and len(val) == len(res_list):
                        return float(val[res_list.index(N)])
                    elif M in res_list and len(val) == len(res_list):
                        return float(val[res_list.index(M)])

        # Check default key in dict
        if "default" in cfg:
            return float(cfg["default"])

    return 1e-12

# ---------------------------------------------------------
# Global Config
# ---------------------------------------------------------
GLOBAL_CONFIG = {
    'num_processors': None,
    'R': 1.0,
    'rad_unif': 1,
    'tol_nufft': 1e-10,
    'maxiter_nufft': 200,
    'reg_param': 'auto',
    'eps_finufft': 1e-12,
    'precond_shift': 1e-3,
    'kde_oversample': 4,
    'kde_bandwidth': 1.0,
    'quad_rule': 1,
    'BC_choice': 1,
    'problem_type': 0,
    'custom_problem': None
}

def set_global_config(**kwargs):
    global GLOBAL_CONFIG, R, RAD_UNIF
    GLOBAL_CONFIG.update(kwargs)
    if 'R' in kwargs: R = kwargs['R']
    if 'rad_unif' in kwargs: RAD_UNIF = kwargs['rad_unif']

def get_global_config():
    return GLOBAL_CONFIG

# ---------------------------------------------------------
# Grid Generation Cache
# ---------------------------------------------------------
ANGLE_CACHE = {}

def get_angle_mesh(method, N, M):
    azu_unif = method["azu_unif"]
    kind = method.get("mesh_kind", "uniform")

    if kind == "uniform":
        return generate_uniform_azimuthal(N)

    if azu_unif == 1:
        key = (kind, N)
        if key not in ANGLE_CACHE:
            if kind in ("rand", "random", "stratified_rand", "stratified_random", "stratified"):
                np.random.seed(42)  # Seed random grid generation for exact reproducibility across solvers
            ANGLE_CACHE[key] = generate_fixed_nonuniform_azimuthal(N, kind=kind)
        return ANGLE_CACHE[key]
    elif azu_unif == 0:
        key = (kind, N, M)
        if key not in ANGLE_CACHE:
            if kind in ("rand", "random", "stratified_rand", "stratified_random", "stratified"):
                np.random.seed(42)
            ANGLE_CACHE[key] = generate_nonuniform_azimuthal(N, M, kind=kind)
        return ANGLE_CACHE[key]
    return generate_uniform_azimuthal(N)

# ---------------------------------------------------------
# Core Single Test Execution
# ---------------------------------------------------------
def run_case(N, M, method, bc_choice=1, quad_rule=1, mute=False, reg_param=None, **kwargs):
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

    cfg_reg = reg_param if reg_param is not None else GLOBAL_CONFIG.get('reg_param', 'auto')
    actual_reg = resolve_reg_param(cfg_reg, method, N, M, iAngle)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        if mute:
            warnings.simplefilter("ignore")

        if "grid_type" in method:
            grid_type = method["grid_type"]
        else:
            if solver_azu == 2:
                grid_type = 1
            elif solver_azu == 1:
                mesh_kind = method.get("mesh_kind", "")
                if mesh_kind in ("jittered", "stratified_rand", "stratified_random", "stratified"):
                    grid_type = 2
                else:
                    grid_type = 3
            else:
                grid_type = solver_azu

        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

        t0 = time.perf_counter()

        try:
            u_approx = poisson_solver(
                f_values, g_values, u_fourier_0,
                N, M, iRadius, iAngle, R,
                quad_rule=quad_rule, BC_choice=bc_choice,
                rad_unif=RAD_UNIF,
                grid_type=grid_type,
                use_nudft_angular=nudft_flag,
                maxiter_nufft=GLOBAL_CONFIG.get('maxiter_nufft', 50), 
                tol_nufft=GLOBAL_CONFIG.get('tol_nufft', 1e-8),
                reg_param=actual_reg,
                eps_finufft=GLOBAL_CONFIG.get('eps_finufft', 1e-12),
                precond_shift=GLOBAL_CONFIG.get('precond_shift', 1e-3),
                kde_oversample=GLOBAL_CONFIG.get('kde_oversample', 4),
                kde_bandwidth=GLOBAL_CONFIG.get('kde_bandwidth', 1.0),
                num_processors=GLOBAL_CONFIG.get('num_processors', None)
            )

            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass

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
def run_tests_pipeline(N_values, M_values, fixed_other, methods, test_type="P1_Table1", mute=False, reg_param=None, **kwargs):
    results = []
    method_pbar = tqdm(methods, desc=f"Pipeline ({test_type})", disable=False)
    for method in method_pbar:
        method_pbar.set_postfix(method=method['name'])
        if not mute:
            print(f"\n{method['label']}")

        if test_type == "P1_Table1":
            pairs = [(N, M) for N in N_values for M in M_values]
            sub_pbar = tqdm(pairs, desc=f"  {method['label']}", leave=False, disable=False)
            for N, M in sub_pbar:
                res = run_case(N, M, method, bc_choice=1, quad_rule=1, mute=mute, reg_param=reg_param, **kwargs)
                results.append(res)
                if not mute:
                    print(f"  N={N:4d}, M={M:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "P1_Table2":
            tasks = [(M, quad, bc) for M in M_values for quad in [1, 2] for bc in [1, 2]]
            sub_pbar = tqdm(tasks, desc=f"  {method['label']}", leave=False, disable=False)
            for M, quad, bc in sub_pbar:
                res = run_case(fixed_other, M, method, bc_choice=bc, quad_rule=quad, mute=mute, reg_param=reg_param, **kwargs)
                results.append(res)
                q_str = "Trapezoidal" if quad == 1 else "Simpson"
                bc_str = "Dirichlet" if bc == 1 else "Neumann"
                if not mute:
                    print(f"  M={M:4d}, {q_str}, {bc_str} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "Accuracy_VaryN":
            sub_pbar = tqdm(N_values, desc=f"  {method['label']}", leave=False, disable=False)
            for N in sub_pbar:
                res = run_case(N, fixed_other, method, bc_choice=1, quad_rule=1, mute=mute, reg_param=reg_param, **kwargs)
                results.append(res)
                if not mute:
                    print(f"  N={N:4d} | L2_rel={res['L2_rel']:.3e} | t={res['runtime']:.3f}s")
        elif test_type == "Accuracy_VaryM":
            sub_pbar = tqdm(M_values, desc=f"  {method['label']}", leave=False, disable=False)
            for M in sub_pbar:
                res = run_case(fixed_other, M, method, bc_choice=1, quad_rule=1, mute=mute, reg_param=reg_param, **kwargs)
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

    formatted_table = pivot.map(fmt)
    
    # Check if we are inside a Jupyter notebook environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            display(HTML(formatted_table.to_html()))
        else:
            print(formatted_table.to_string())
    except Exception:
        print(formatted_table.to_string())


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
# Standalone Plotting Helpers (Accuracy & Runtime Visualizations)
# ---------------------------------------------------------
def plot_accuracy_table1(df, title_prefix="Table 1"):
    """
    Generate 6 subplots (2x3 grid) for Table 1 Accuracy:
    - Top row (3 subplots): Error vs M for FFT, NUDFT, NUFFT
    - Bottom row (3 subplots): Error vs N for FFT, NUDFT, NUFFT
    """
    methods = df["label"].unique()
    m_fft = [m for m in methods if "FFT" in m and "NUFFT" not in m]
    m_nudft = [m for m in methods if "NUDFT" in m]
    m_nufft_t = [m for m in methods if "NUFFT" in m and "Toeplitz" in m]
    m_nufft_u = [m for m in methods if "NUFFT" in m and "Unsquared" in m]
    m_nufft_gen = [m for m in methods if "NUFFT" in m and "Toeplitz" not in m and "Unsquared" not in m]
    
    solver_groups = []
    if m_fft: solver_groups.append(("Uniform / FFT", m_fft))
    if m_nudft: solver_groups.append(("NUDFT", m_nudft))
    if m_nufft_gen: solver_groups.append(("NUFFT", m_nufft_gen))
    if m_nufft_t: solver_groups.append(("NUFFT (Toeplitz)", m_nufft_t))
    if m_nufft_u: solver_groups.append(("NUFFT (Unsquared)", m_nufft_u))
    
    num_cols = len(solver_groups)
    fig, axes = plt.subplots(2, num_cols, figsize=(4.5 * num_cols, 5), sharey=True)
    if num_cols == 1: axes = axes.reshape(2, 1)
    
    # Top Row: Error vs M (for each N)
    for col_idx, (s_name, s_methods) in enumerate(solver_groups):
        ax = axes[0, col_idx]
        df_sub = df[df["label"].isin(s_methods)]
        for N, group in df_sub.groupby("N"):
            ax.loglog(group["M"], group["L2_rel"], marker="o", label=f"N={N}")
        ax.set_xlabel("Radial Grid Points (M)")
        ax.set_ylabel("Relative $L_2$ Error")
        ax.set_title(f"{title_prefix} - {s_name} (Error vs M)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    # Bottom Row: Error vs N (for each M)
    for col_idx, (s_name, s_methods) in enumerate(solver_groups):
        ax = axes[1, col_idx]
        df_sub = df[df["label"].isin(s_methods)]
        for M, group in df_sub.groupby("M"):
            ax.loglog(group["N"], group["L2_rel"], marker="s", label=f"M={M}")
        ax.set_xlabel("Angular Grid Points (N)")
        ax.set_ylabel("Relative $L_2$ Error")
        ax.set_title(f"{title_prefix} - {s_name} (Error vs N)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_runtime_table1(df, title_prefix="Table 1"):
    """
    Generate 2xN grid for Table 1 Runtime:
    - Top row: Runtime vs M, one subplot for each N. Each plot has the 3 methods.
    - Bottom row: Runtime vs N, one subplot for each M. Each plot has the 3 methods.
    """
    N_vals = sorted(df["N"].unique())
    M_vals = sorted(df["M"].unique())
    
    num_cols = max(len(N_vals), len(M_vals))
    
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8), sharey=True)
    
    # Ensure axes is 2D even if num_cols is 1
    if num_cols == 1:
        axes = axes.reshape(2, 1)
    
    # Top Row: Runtime vs M for each N
    for i, N in enumerate(N_vals):
        ax = axes[0, i]
        df_sub = df[df["N"] == N]
        for label, group in df_sub.groupby("label"):
            group = group.sort_values("M")
            ax.loglog(group["M"], group["runtime"], marker="o", label=label)
        ax.set_xlabel("Radial Grid Points (M)")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"{title_prefix} (N={N}) - Runtime vs M")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        
    for i in range(len(N_vals), num_cols):
        axes[0, i].axis('off')

    # Bottom Row: Runtime vs N for each M
    for i, M in enumerate(M_vals):
        ax = axes[1, i]
        df_sub = df[df["M"] == M]
        for label, group in df_sub.groupby("label"):
            group = group.sort_values("N")
            ax.loglog(group["N"], group["runtime"], marker="s", label=label)
        ax.set_xlabel("Angular Grid Points (N)")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"{title_prefix} (M={M}) - Runtime vs N")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        
    for i in range(len(M_vals), num_cols):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_accuracy_table2(df, title_prefix="Table 2"):
    """
    Generate 4 subplots (2x2 grid) for Table 2 Accuracy:
    (Trapezoidal/Dirichlet, Trapezoidal/Neumann, Simpson/Dirichlet, Simpson/Neumann)
    comparing FFT, NUDFT, and NUFFT vs M.
    """
    df_fmt = _prepare_table2_df(df)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    combos = [
        ("Trapezoidal", "Dirichlet", axes[0, 0]),
        ("Trapezoidal", "Neumann", axes[0, 1]),
        ("Simpson", "Dirichlet", axes[1, 0]),
        ("Simpson", "Neumann", axes[1, 1]),
    ]
    for q_str, bc_str, ax in combos:
        df_sub = df_fmt[(df_fmt["quad_str"] == q_str) & (df_fmt["bc_str"] == bc_str)]
        for label, group in df_sub.groupby("label"):
            ax.loglog(group["M"], group["L2_rel"], marker="o", label=label)
        ax.set_xlabel("Radial Grid Points (M)")
        ax.set_ylabel("Relative $L_2$ Error")
        ax.set_title(f"{title_prefix} - {q_str}, {bc_str}")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_runtime_table2(df, title_prefix="Table 2"):
    """
    Generate 4 subplots (2x2 grid) for Table 2 Runtime:
    (Trapezoidal/Dirichlet, Trapezoidal/Neumann, Simpson/Dirichlet, Simpson/Neumann)
    comparing FFT, NUDFT, and NUFFT vs M.
    """
    df_fmt = _prepare_table2_df(df)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    combos = [
        ("Trapezoidal", "Dirichlet", axes[0, 0]),
        ("Trapezoidal", "Neumann", axes[0, 1]),
        ("Simpson", "Dirichlet", axes[1, 0]),
        ("Simpson", "Neumann", axes[1, 1]),
    ]
    for q_str, bc_str, ax in combos:
        df_sub = df_fmt[(df_fmt["quad_str"] == q_str) & (df_fmt["bc_str"] == bc_str)]
        for label, group in df_sub.groupby("label"):
            ax.loglog(group["M"], group["runtime"], marker="s", label=label)
        ax.set_xlabel("Radial Grid Points (M)")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"{title_prefix} - {q_str}, {bc_str}")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(df, index_col="N", title_prefix="Accuracy Comparison"):
    """
    Generate 3 subplots (1x3 grid) for Accuracy Comparison:
    Subplot 1: FFT methods (All meshes)
    Subplot 2: NUDFT methods (All meshes)
    Subplot 3: NUFFT methods (All meshes)
    """
    methods = df["label"].unique()
    m_fft = [m for m in methods if "FFT" in m and "NUFFT" not in m]
    m_nudft = [m for m in methods if "NUDFT" in m]
    m_nufft_t = [m for m in methods if "NUFFT" in m and "Toeplitz" in m]
    m_nufft_u = [m for m in methods if "NUFFT" in m and "Unsquared" in m]
    m_nufft_gen = [m for m in methods if "NUFFT" in m and "Toeplitz" not in m and "Unsquared" not in m]
    
    groups = []
    if m_fft: groups.append(("Uniform / FFT", m_fft))
    if m_nudft: groups.append(("NUDFT", m_nudft))
    if m_nufft_gen: groups.append(("NUFFT", m_nufft_gen))
    if m_nufft_t: groups.append(("NUFFT (Toeplitz)", m_nufft_t))
    if m_nufft_u: groups.append(("NUFFT (Unsquared)", m_nufft_u))
    
    num_cols = len(groups)
    fig, axes = plt.subplots(1, num_cols, figsize=(4.5 * num_cols, 3), sharey=True)
    if num_cols == 1: axes = [axes]
    
    solver_groups = [(g[0], g[1], axes[i]) for i, g in enumerate(groups)]
    
    for s_name, s_methods, ax in solver_groups:
        df_sub = df[df["label"].isin(s_methods)]
        for label, group in df_sub.groupby("label"):
            ax.loglog(group[index_col], group["L2_rel"], marker="o", label=label)
        ax.set_xlabel(f"{index_col} Grid Points")
        ax.set_ylabel("Relative $L_2$ Error")
        ax.set_title(f"{title_prefix} - {s_name}")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_runtime_comparison(df, index_col="N", title_prefix="Accuracy Comparison"):
    """
    Generate 3 subplots (1x3 grid) for Runtime Comparison:
    Subplot 1: FFT methods (All meshes)
    Subplot 2: NUDFT methods (All meshes)
    Subplot 3: NUFFT methods (All meshes)
    """
    methods = df["label"].unique()
    m_fft = [m for m in methods if "FFT" in m and "NUFFT" not in m]
    m_nudft = [m for m in methods if "NUDFT" in m]
    m_nufft_t = [m for m in methods if "NUFFT" in m and "Toeplitz" in m]
    m_nufft_u = [m for m in methods if "NUFFT" in m and "Unsquared" in m]
    m_nufft_gen = [m for m in methods if "NUFFT" in m and "Toeplitz" not in m and "Unsquared" not in m]
    
    groups = []
    if m_fft: groups.append(("Uniform / FFT", m_fft))
    if m_nudft: groups.append(("NUDFT", m_nudft))
    if m_nufft_gen: groups.append(("NUFFT", m_nufft_gen))
    if m_nufft_t: groups.append(("NUFFT (Toeplitz)", m_nufft_t))
    if m_nufft_u: groups.append(("NUFFT (Unsquared)", m_nufft_u))
    
    num_cols = len(groups)
    fig, axes = plt.subplots(1, num_cols, figsize=(4.5 * num_cols, 4), sharey=True)
    if num_cols == 1: axes = [axes]
    
    solver_groups = [(g[0], g[1], axes[i]) for i, g in enumerate(groups)]
    
    for s_name, s_methods, ax in solver_groups:
        df_sub = df[df["label"].isin(s_methods)]
        for label, group in df_sub.groupby("label"):
            ax.loglog(group[index_col], group["runtime"], marker="s", label=label)
        ax.set_xlabel(f"{index_col} Grid Points")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"{title_prefix} - {s_name}")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

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

            solver_azu = method.get("solver_azu_unif", method["azu_unif"])
            if "grid_type" in method:
                grid_type = method["grid_type"]
            else:
                if solver_azu == 2:
                    grid_type = 1
                elif solver_azu == 1:
                    mesh_kind = method.get("mesh_kind", "")
                    if mesh_kind in ("jittered", "stratified_rand", "stratified_random", "stratified"):
                        grid_type = 2
                    else:
                        grid_type = 3
                else:
                    grid_type = solver_azu

            try:
                u_approx = poisson_solver(
                    f_values, g_values, u_fourier_0, N_fixed, M, iRadius, iAngle, R,
                    quad_rule=quad_rule, BC_choice=bc_choice, rad_unif=RAD_UNIF, grid_type=grid_type,
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

    actual_reg = resolve_reg_param(GLOBAL_CONFIG.get('reg_param', 'auto'), method, N, M, iAngle)

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
                maxiter_nufft=GLOBAL_CONFIG['maxiter_nufft'], 
                tol_nufft=GLOBAL_CONFIG['tol_nufft'],
                reg_param=actual_reg,
                eps_finufft=GLOBAL_CONFIG.get('eps_finufft', 1e-12),
                precond_shift=GLOBAL_CONFIG.get('precond_shift', 1e-3),
                kde_oversample=GLOBAL_CONFIG.get('kde_oversample', 4),
                kde_bandwidth=GLOBAL_CONFIG.get('kde_bandwidth', 1.0)
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




def plot_runtime_table1_extremes(df, title_prefix="Table 1 (extreme N,M)"):
    """
    2x2 runtime plot using only the lowest and highest N and M:
      Top-left:  runtime vs M for N = N_min
      Top-right: runtime vs M for N = N_max
      Bottom-left:  runtime vs N for M = M_min
      Bottom-right: runtime vs N for M = M_max
    All solver labels are plotted in each relevant panel.
    """
    N_vals = sorted(df["N"].unique())
    M_vals = sorted(df["M"].unique())
    if len(N_vals) < 2 or len(M_vals) < 2:
        print("Need at least two distinct N and M values for extremes plot.")
        return

    N_lo, N_hi = N_vals[0], N_vals[-1]
    M_lo, M_hi = M_vals[0], M_vals[-1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")
    fig.suptitle(title_prefix)

    # Top-left: runtime vs M for N = N_lo
    ax = axes[0, 0]
    df_sub = df[df["N"] == N_lo]
    for label, group in df_sub.groupby("label"):
        group = group.sort_values("M")
        ax.loglog(group["M"], group["runtime"], marker="o", label=label)
    ax.set_xlabel("Radial Grid Points (M)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"N = {N_lo} — Runtime vs M")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)

    # Top-right: runtime vs M for N = N_hi
    ax = axes[0, 1]
    df_sub = df[df["N"] == N_hi]
    for label, group in df_sub.groupby("label"):
        group = group.sort_values("M")
        ax.loglog(group["M"], group["runtime"], marker="o", label=label)
    ax.set_xlabel("Radial Grid Points (M)")
    ax.set_title(f"N = {N_hi} — Runtime vs M")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Bottom-left: runtime vs N for M = M_lo
    ax = axes[1, 0]
    df_sub = df[df["M"] == M_lo]
    for label, group in df_sub.groupby("label"):
        group = group.sort_values("N")
        ax.loglog(group["N"], group["runtime"], marker="s", label=label)
    ax.set_xlabel("Angular Grid Points (N)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"M = {M_lo} — Runtime vs N")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)

    # Bottom-right: runtime vs N for M = M_hi
    ax = axes[1, 1]
    df_sub = df[df["M"] == M_hi]
    for label, group in df_sub.groupby("label"):
        group = group.sort_values("N")
        ax.loglog(group["N"], group["runtime"], marker="s", label=label)
    ax.set_xlabel("Angular Grid Points (N)")
    ax.set_title(f"M = {M_hi} — Runtime vs N")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()