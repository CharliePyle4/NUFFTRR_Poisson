import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, HTML
from tqdm.auto import tqdm

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Poisson_Solver.grids import (
    generate_uniform_azimuthal,
    generate_cartesian_grid_on_disk,
    compute_zero_mode,
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver


# ==============================================================================
# Manufactured Problem Definitions
# ==============================================================================

def get_boundary_layer_problem(R=1.0, mode=4, lambda_param=5.0, theta_0=0.0):
    """
    Manufactured Poisson problem with a sharp outer boundary layer at r -> R:
        u(r, theta) = (1 - exp(lambda * (r^2/R^2 - 1))) * (r/R)^m * cos(m (theta - theta_0))
        u(R, theta) = 0 (Homogeneous Dirichlet)
    """
    m = int(mode)
    lam = float(lambda_param)

    def u(x, y):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        q = (r / R) ** m
        E = np.exp(lam * (r**2 / R**2 - 1.0))
        c = np.cos(m * (theta - theta_0))
        return (1.0 - E) * q * c

    def f(x, y):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        q = (r / R) ** m
        E = np.exp(lam * (r**2 / R**2 - 1.0))
        c = np.cos(m * (theta - theta_0))
        term = (4.0 * lam / R**2) * (m + 1.0 + lam * (r**2 / R**2))
        return -E * q * term * c

    def g_dir(x, y):
        return np.zeros_like(x)

    def g_neu(x, y):
        theta = np.arctan2(y, x)
        c = np.cos(m * (theta - theta_0))
        # du/dr at r=R: (1-1)*q' - E'(R)*q = -2*lam/R * c
        return (-2.0 * lam / R) * c

    return {
        "name": f"Boundary Layer (lambda={lam})",
        "u": u,
        "f": f,
        "g_dirichlet": g_dir,
        "g_neumann": g_neu,
        "R": R,
        "mode": m,
        "lambda": lam,
        "theta_0": theta_0,
    }


def get_core_concentrated_problem(R=1.0, mode=1, alpha_param=25.0, theta_0=0.0):
    """
    Manufactured Poisson problem with origin core concentration at r -> 0:
        u(r, theta) = (R^2 - r^2) * exp(-alpha * r^2/R^2) * (r/R)^m * cos(m (theta - theta_0))
        u(R, theta) = 0 (Homogeneous Dirichlet)
    """
    m = int(mode)
    alpha = float(alpha_param)

    def u(x, y):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        q = (r / R) ** m
        E = np.exp(-alpha * r**2 / R**2)
        c = np.cos(m * (theta - theta_0))
        return (R**2 - r**2) * E * q * c

    def f(x, y):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        r_R = r / R
        r_R2 = r_R**2
        q = r_R**m
        E = np.exp(-alpha * r_R2)
        c = np.cos(m * (theta - theta_0))

        bracket = (
            -(m + 1.0) * (alpha + 1.0)
            + alpha * (alpha + m + 3.0) * r_R2
            - (alpha**2) * (r_R2**2)
        )
        return 4.0 * E * q * bracket * c

    def g_dir(x, y):
        return np.zeros_like(x)

    def g_neu(x, y):
        theta = np.arctan2(y, x)
        c = np.cos(m * (theta - theta_0))
        E_R = np.exp(-alpha)
        # du/dr at r=R: (-2R)*E_R*1*c
        return -2.0 * R * E_R * c

    return {
        "name": f"Core Concentration (alpha={alpha})",
        "u": u,
        "f": f,
        "g_dirichlet": g_dir,
        "g_neumann": g_neu,
        "R": R,
        "mode": m,
        "alpha": alpha,
        "theta_0": theta_0,
    }


# ==============================================================================
# Radial Mesh Generation
# ==============================================================================

def generate_custom_radial_grid(M, R=1.0, kind="uniform", **kwargs):
    """
    Generate 1D radial grid r_m on [0, R].

    Kinds:
        'uniform': Equispaced spacing
        'sinh': Outer boundary layer stretching (gamma)
        'sqrt': Outer rim clustering
        'cubic_root': Stronger outer rim clustering
        'squared': Core concentration near r=0
        'cubic': Stronger core concentration near r=0
        'chebyshev_lobatto': Dual boundary/core clustering
    """
    xi = np.linspace(0.0, 1.0, M)

    if kind in ("uniform", "equispaced", "linear"):
        return R * xi
    elif kind in ("sinh", "sinh_boundary"):
        gamma = kwargs.get("gamma", 4.5)
        return R * (1.0 - np.sinh(gamma * (1.0 - xi)) / np.sinh(gamma))
    elif kind == "sinh_core":
        gamma = kwargs.get("gamma", 2.5)
        return R * np.sinh(gamma * xi) / np.sinh(gamma)
    elif kind == "sqrt":
        # Cluster near r=R
        return R * (1.0 - (1.0 - xi)**2)
    elif kind == "cubic_root":
        return R * (1.0 - (1.0 - xi)**3)
    elif kind == "squared":
        return R * (xi**2)
    elif kind == "cubic":
        return R * (xi**3)
    elif kind in ("chebyshev_lobatto", "chebyshev", "lobatto"):
        return 0.5 * R * (1.0 - np.cos(np.pi * xi))
    else:
        raise ValueError(f"Unknown radial mesh kind: '{kind}'")


# ==============================================================================
# Multi-Run Timing Configuration
# ==============================================================================
# ==============================================================================
# Multi-Run Timing Configuration & Global Backend
# ==============================================================================
TIME_TRIALS = False  # Set to True to run each solve 5 times and record min runtime
NUM_RUNS = 5
GLOBAL_USE_GPU = False

def set_timing_config(time_trials=False, num_runs=5, use_gpu=False):
    """Globally configure multi-trial benchmark timing and default backend."""
    global TIME_TRIALS, NUM_RUNS, GLOBAL_USE_GPU
    TIME_TRIALS = bool(time_trials)
    NUM_RUNS = int(num_runs) if time_trials else 1
    GLOBAL_USE_GPU = bool(use_gpu)


# ==============================================================================
# Helper Functions for Radial Labels and Ordering
# ==============================================================================

def normalize_radial_label(label: str) -> str:
    """
    Normalize radial grid labels so they never display 'Adapted'.
    Example:
        'Adapted Nonuniform (Squared)' -> 'Nonuniform (Squared)'
        'Adapted Nonuniform (Sinh)'    -> 'Nonuniform (Sinh)'
        'Adapted Core Radial (Squared)'-> 'Nonuniform (Squared)'
        'Adapted Boundary Radial (Sinh)'-> 'Nonuniform (Sinh)'
    """
    if not isinstance(label, str):
        return label
    cleaned = label.replace("Adapted ", "").replace("Adapted / ", "")
    cleaned = cleaned.replace("Core Radial (Squared)", "Nonuniform (Squared)")
    cleaned = cleaned.replace("Boundary Radial (Sinh)", "Nonuniform (Sinh)")
    return cleaned


def sort_radial_columns(columns):
    """
    Sort radial grid labels so that the ordering is:
    1. Uniform Radial
    2. Chebyshev-Lobatto
    3. Nonuniform (Squared, Sinh, etc.)
    """
    def sort_key(col):
        c = str(col).lower()
        if "uniform" in c and "non" not in c:
            return (0, c)
        elif "chebyshev" in c:
            return (1, c)
        elif "nonuniform" in c or "non-uniform" in c:
            return (2, c)
        else:
            return (3, c)
    return sorted(columns, key=sort_key)


def run_radial_benchmark_case(
    N,
    M,
    r_m,
    problem,
    rad_kind="custom",
    bc_choice=1,
    quad_rule=2,
    num_processors=None,
    use_gpu=None,
    time_trials=None,
    num_runs=None,
    **kwargs,
):
    """
    Execute Poisson solve on uniform theta and specified radial grid r_m.

    rad_unif convention:
        rad_unif = 1 for uniform radial grid
        rad_unif = 0 for non-uniform radial grid
    """
    actual_use_gpu = GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)
    effective_time_trials = TIME_TRIALS if time_trials is None else bool(time_trials)
    R = problem["R"]
    theta_solver = generate_uniform_azimuthal(N)

    # 1. Evaluate analytical forcing on disk grid
    x_grid, y_grid = generate_cartesian_grid_on_disk(theta_solver, r_m)
    f_values = problem["f"](x_grid, y_grid)

    if bc_choice == 1:
        g_values = problem["g_dirichlet"](x_grid[:, -1], y_grid[:, -1])
        u_fourier_0 = 0.0
    else:
        g_values = problem["g_neumann"](x_grid[:, -1], y_grid[:, -1])
        u_exact_grid = problem["u"](x_grid, y_grid)
        u_fourier_0 = compute_zero_mode(u_exact_grid, theta_solver, azu_unif=2)[-1]

    # Detect if grid is strictly uniform
    dr = np.diff(r_m)
    is_uniform = np.allclose(dr, dr[0], rtol=1e-8, atol=1e-12)
    rad_unif_flag = 1 if is_uniform else 0

    # Pre-convert inputs to GPU device memory before the timing loop
    if actual_use_gpu:
        try:
            import cupy as cp
            f_in = cp.asarray(f_values)
            g_in = cp.asarray(g_values)
            r_in = cp.asarray(r_m)
            th_in = cp.asarray(theta_solver)
            u0_in = cp.asarray(u_fourier_0) if u_fourier_0 is not None else 0.0
            cp.cuda.Stream.null.synchronize()
        except Exception:
            f_in, g_in, r_in, th_in, u0_in = f_values, g_values, r_m, theta_solver, u_fourier_0
    else:
        f_in, g_in, r_in, th_in, u0_in = f_values, g_values, r_m, theta_solver, u_fourier_0

    # 2. Timed Poisson Solve
    n_runs = num_runs if num_runs is not None else (NUM_RUNS if effective_time_trials else 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        runtimes = []
        u_approx = None
        for _ in range(n_runs):
            if actual_use_gpu:
                try:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()
                except Exception:
                    pass

            t0 = time.perf_counter()

            u_approx = poisson_solver(
                f_values=f_in,
                g_values=g_in,
                u_fourier_0=u0_in,
                N=N,
                M=M,
                r_m=r_in,
                theta_j=th_in,
                R=R,
                quad_rule=quad_rule,
                BC_choice=bc_choice,
                rad_unif=rad_unif_flag,
                grid_type=1,  # Uniform FFT in theta
                num_processors=num_processors,
                use_gpu=actual_use_gpu,
                **kwargs,
            )

            if actual_use_gpu:
                try:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()
                except Exception:
                    pass

            runtimes.append(time.perf_counter() - t0)

        runtime = min(runtimes)

    # 3. Exact Solution and Error Metrics
    if actual_use_gpu and hasattr(u_approx, "get"):
        u_approx = u_approx.get()

    u_true = problem["u"](x_grid, y_grid)
    _, linf_rel, _, l2_rel = compute_error_metrics(
        u_approx, u_true, r_m, theta_solver
    )

    return {
        "N": N,
        "M": M,
        "rad_kind": normalize_radial_label(rad_kind),
        "is_uniform": is_uniform,
        "L2_rel": l2_rel,
        "Linf_rel": linf_rel,
        "runtime": runtime,
        "u_approx": u_approx,
        "u_true": u_true,
        "x_coord": x_grid,
        "y_coord": y_grid,
        "r_m": r_m,
        "theta_j": theta_solver,
    }


def run_radial_grid_sweep(
    problem,
    N,
    M_values,
    rad_kinds,
    quad_rule=2,
    bc_choice=1,
    num_processors=None,
    use_gpu=None,
    time_trials=None,
    num_runs=None,
    **kwargs,
):
    """
    Run M-refinement study across multiple radial mesh choices for a fixed N.
    """
    actual_use_gpu = GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)
    # Dummy Warmup Solve (Warms up thread pools, CPU cache, and GPU plans)
    try:
        if M_values and rad_kinds:
            r_warmup = generate_custom_radial_grid(M_values[0], R=problem["R"], kind=rad_kinds[0][0])
            run_radial_benchmark_case(
                N=N,
                M=M_values[0],
                r_m=r_warmup,
                problem=problem,
                rad_kind=rad_kinds[0][1],
                bc_choice=bc_choice,
                quad_rule=quad_rule,
                num_processors=num_processors,
                use_gpu=actual_use_gpu,
                time_trials=False,
                num_runs=1,
                **kwargs,
            )
    except Exception:
        pass

    rows = []
    backend_label = "GPU" if actual_use_gpu else "CPU"
    pbar = tqdm(
        total=len(M_values) * len(rad_kinds),
        desc=f"Radial M-sweep ({problem['name']}) [{backend_label}]",
    )
    for M in M_values:
        for kind, label in rad_kinds:
            r_m = generate_custom_radial_grid(M, R=problem["R"], kind=kind)
            res = run_radial_benchmark_case(
                N=N,
                M=M,
                r_m=r_m,
                problem=problem,
                rad_kind=label,
                bc_choice=bc_choice,
                quad_rule=quad_rule,
                num_processors=num_processors,
                use_gpu=actual_use_gpu,
                time_trials=time_trials,
                num_runs=num_runs,
                **kwargs,
            )
            rows.append(res)
            pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    if not df.empty and "rad_kind" in df.columns:
        df["rad_kind"] = df["rad_kind"].apply(normalize_radial_label)
    return df


def run_nxm_grid_sweep(
    problem,
    N_values,
    M_values,
    rad_kinds,
    quad_rule=2,
    bc_choice=1,
    num_processors=None,
    use_gpu=None,
    time_trials=None,
    num_runs=None,
    **kwargs,
):
    """
    Run full N x M grid sweep across angular counts N and radial counts M.
    """
    actual_use_gpu = GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)
    # Dummy Warmup Solve
    try:
        if N_values and M_values and rad_kinds:
            r_warmup = generate_custom_radial_grid(M_values[0], R=problem["R"], kind=rad_kinds[0][0])
            run_radial_benchmark_case(
                N=N_values[0],
                M=M_values[0],
                r_m=r_warmup,
                problem=problem,
                rad_kind=rad_kinds[0][1],
                bc_choice=bc_choice,
                quad_rule=quad_rule,
                num_processors=num_processors,
                use_gpu=actual_use_gpu,
                time_trials=False,
                num_runs=1,
                **kwargs,
            )
    except Exception:
        pass

    rows = []
    backend_label = "GPU" if actual_use_gpu else "CPU"
    pbar = tqdm(
        total=len(N_values) * len(M_values) * len(rad_kinds),
        desc=f"N x M Sweep ({problem['name']}) [{backend_label}]",
    )
    for N in N_values:
        for M in M_values:
            for kind, label in rad_kinds:
                r_m = generate_custom_radial_grid(M, R=problem["R"], kind=kind)
                res = run_radial_benchmark_case(
                    N=N,
                    M=M,
                    r_m=r_m,
                    problem=problem,
                    rad_kind=label,
                    bc_choice=bc_choice,
                    quad_rule=quad_rule,
                    num_processors=num_processors,
                    use_gpu=actual_use_gpu,
                    time_trials=time_trials,
                    num_runs=num_runs,
                    **kwargs,
                )
                rows.append(res)
                pbar.update(1)
    pbar.close()
    df = pd.DataFrame(rows)
    if not df.empty and "rad_kind" in df.columns:
        df["rad_kind"] = df["rad_kind"].apply(normalize_radial_label)
    return df




# ==============================================================================
# Visualization & Table Formatting Routines
# ==============================================================================

def plot_solution_and_radial_grids(problem, N=32, M=16, primary_kind="sinh", primary_label="Nonuniform Radial"):
    """
    Journal-style 1 x 3 visualization:
      (a) Exact solution 3D surface plot on disk,
      (b) Uniform radial polar grid,
      (c) Non-uniform radial polar grid.
    """
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
    })

    R = problem["R"]
    theta_grid = generate_uniform_azimuthal(N)
    r_unif = generate_custom_radial_grid(M, R=R, kind="uniform")
    r_nonunif = generate_custom_radial_grid(M, R=R, kind=primary_kind)

    fine_theta = np.linspace(0, 2 * np.pi, 250, endpoint=False)
    fine_r = np.linspace(0, R, 150)
    Xf, Yf = generate_cartesian_grid_on_disk(fine_theta, fine_r)
    Uf = problem["u"](Xf, Yf)

    fig = plt.figure(figsize=(12, 3.8), dpi=150)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.22, 1, 1],
        left=0.045, right=0.985,
        bottom=0.12, top=0.84,
        wspace=0.20,
    )

    # Subplot (a): 3D Exact Solution Surface
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    surf = ax0.plot_surface(
        Xf, Yf, Uf,
        cmap="plasma",
        edgecolor="none",
        antialiased=True,
        rcount=120,
        ccount=180,
    )
    ax0.set_title(f"(a) Exact Solution: {problem['name']}", fontsize=8, pad=8, fontweight="semibold")
    ax0.set_xlabel(r"$x$", fontsize=8, labelpad=2)
    ax0.set_ylabel(r"$y$", fontsize=8, labelpad=2)
    ax0.set_zlabel("")
    ax0.set_xlim(-R, R)
    ax0.set_ylim(-R, R)
    ax0.view_init(elev=27, azim=-52)
    ax0.set_box_aspect((1, 1, 0.58), zoom=1.00)
    ax0.set_xticks([-R, -R/2, 0, R/2, R])
    ax0.set_yticks([-R, -R/2, 0, R/2, R])
    ax0.set_zticks([])
    ax0.tick_params(axis="x", labelsize=8, pad=1)
    ax0.tick_params(axis="y", labelsize=8, pad=1)

    cbar = fig.colorbar(surf, ax=ax0, shrink=0.62, aspect=22, pad=0.14)
    cbar.set_label(r"$u(x,y)$", rotation=270, labelpad=8, fontsize=8)
    cbar.ax.tick_params(labelsize=8, pad=2)

    # Helper for 2D polar grid panels
    def draw_grid_lines(ax_obj, r_vec, title_text, col):
        X, Y = generate_cartesian_grid_on_disk(theta_grid, r_vec)
        for r_val in r_vec[1:]:
            ax_obj.add_patch(Circle((0, 0), r_val, fill=False, ec="0.85", lw=0.4))
        for th in theta_grid:
            ax_obj.plot([0, R * np.cos(th)], [0, R * np.sin(th)], color="0.85", lw=0.4)
        ax_obj.scatter(X, Y, s=4, color=col, zorder=3)
        ax_obj.add_patch(Circle((0, 0), R, fill=False, ec="black", lw=1.0))
        ax_obj.set_title(title_text, fontsize=8, fontweight="semibold")
        ax_obj.set_aspect("equal")
        ax_obj.axis("off")

    # Subplot (b): Uniform Radial Grid
    ax1 = fig.add_subplot(gs[0, 1])
    draw_grid_lines(ax1, r_unif, f"(b) Uniform Radial Grid (M={M})", "#1f77b4")

    # Subplot (c): Nonuniform Radial Grid
    ax2 = fig.add_subplot(gs[0, 2])
    draw_grid_lines(ax2, r_nonunif, f"(c) {normalize_radial_label(primary_label)} (M={M})", "#d62728")

    plt.show()


def plot_combined_radial_convergence_and_profiles(
    df_fixed_n, results_list, title="Radial Grid Performance & Error Profiles"
):
    """
    Journal-style 1 x 2 figure:
      (a) Log-Log Radial M-Refinement Convergence (Linf relative error vs M),
      (b) Semilog-Y Pointwise Error Profile along Radial Ray r in [0, R].
    """
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), dpi=150)

    styles = {
        "Uniform Radial": ("#1f77b4", "o", "-"),
        "Chebyshev-Lobatto": ("#2ca02c", "^", "-."),
        "Nonuniform (Sinh)": ("#d62728", "s", "--"),
        "Nonuniform (Squared)": ("#d62728", "s", "--"),
        "Adapted Nonuniform (Sinh)": ("#d62728", "s", "--"),
        "Adapted Nonuniform (Squared)": ("#d62728", "s", "--"),
    }

    # Subplot (a): M-Refinement Convergence (Log-Log)
    ax0 = axes[0]
    df_plot = df_fixed_n.copy()
    df_plot["rad_kind"] = df_plot["rad_kind"].apply(normalize_radial_label)
    methods_order = sort_radial_columns(df_plot["rad_kind"].unique())
    for method in methods_order:
        grp = df_plot[df_plot["rad_kind"] == method]
        color, marker, ls = styles.get(method, ("black", "d", ":"))
        grp_sorted = grp.sort_values("M")
        ax0.loglog(
            grp_sorted["M"],
            grp_sorted["Linf_rel"],
            label=method,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=1.2,
            markersize=4,
        )

    ax0.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax0.set_xlabel("Radial Points ($M$)")
    ax0.set_ylabel("Relative $L_\\infty$ Error")
    ax0.set_title("(a) Radial $M$-Refinement Convergence")
    ax0.legend(frameon=True, facecolor="white", edgecolor="0.8")

    # Subplot (b): Pointwise Error Profiles along Radial Ray r (Semilog-Y)
    ax1 = axes[1]
    M_eval = results_list[0]["M"]
    sorted_results = sorted(
        results_list,
        key=lambda r: (
            0 if ("uniform" in str(r["rad_kind"]).lower() and "non" not in str(r["rad_kind"]).lower())
            else (1 if "chebyshev" in str(r["rad_kind"]).lower() else 2)
        )
    )
    for res in sorted_results:
        label = normalize_radial_label(res["rad_kind"])
        r_m = res["r_m"]
        color, marker, ls = styles.get(label, ("black", "d", ":"))
        err_ray = np.abs(res["u_approx"][0, :] - res["u_true"][0, :])
        ax1.semilogy(
            r_m,
            err_ray + 1e-16,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.2,
        )

    ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Radial Distance $r$")
    ax1.set_ylabel("Pointwise Error $|u_{approx}(r) - u_{true}(r)|$")
    ax1.set_title(f"(b) Pointwise Radial Error Profile ($M = {M_eval}$)")
    ax1.legend(frameon=True, facecolor="white", edgecolor="0.8")

    fig.suptitle(title, y=1.02, fontweight="semibold")
    plt.tight_layout()
    plt.show()


def plot_m_refinement_convergence(df_results, title="Radial M-Refinement Convergence"):
    """
    Plot Linf relative error vs M for radial grid choices.
    """
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=150)
    styles = {
        "Uniform Radial": ("#1f77b4", "o", "-"),
        "Chebyshev-Lobatto": ("#2ca02c", "^", "-."),
        "Nonuniform (Sinh)": ("#d62728", "s", "--"),
        "Nonuniform (Squared)": ("#d62728", "s", "--"),
        "Adapted Nonuniform (Sinh)": ("#d62728", "s", "--"),
        "Adapted Nonuniform (Squared)": ("#d62728", "s", "--"),
    }

    df_plot = df_results.copy()
    df_plot["rad_kind"] = df_plot["rad_kind"].apply(normalize_radial_label)
    methods_order = sort_radial_columns(df_plot["rad_kind"].unique())
    for method in methods_order:
        grp = df_plot[df_plot["rad_kind"] == method]
        color, marker, ls = styles.get(method, ("black", "d", ":"))
        grp_sorted = grp.sort_values("M")
        ax.loglog(
            grp_sorted["M"],
            grp_sorted["Linf_rel"],
            label=method,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=1.2,
            markersize=5,
        )

    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Number of Radial Rings (M)")
    ax.set_ylabel("Relative $L_\\infty$ Error")
    ax.set_title(title)
    ax.legend(frameon=True, facecolor="white", edgecolor="0.8")
    plt.tight_layout()
    plt.show()


def plot_nxm_accuracy_grid(df_results, title="N x M Convergence Sweep"):
    """
    Faceted plot: Error vs M for different N values across radial grid methods.
    """
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    df_plot = df_results.copy()
    df_plot["rad_kind"] = df_plot["rad_kind"].apply(normalize_radial_label)
    methods = sort_radial_columns(df_plot["rad_kind"].unique())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5), sharey=True, dpi=150)
    if n_methods == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    N_vals = sorted(df_plot["N"].unique())

    for idx, method in enumerate(methods):
        ax = axes[idx]
        sub = df_plot[df_plot["rad_kind"] == method]
        for i, N in enumerate(N_vals):
            grp = sub[sub["N"] == N].sort_values("M")
            ax.loglog(grp["M"], grp["Linf_rel"], "o-", color=cmap(i), label=f"N={N}", markersize=4, linewidth=1.2)
        ax.set_title(f"{method}")
        ax.set_xlabel("Radial Points (M)")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.legend(frameon=True, facecolor="white", edgecolor="0.8")

    axes[0].set_ylabel("Relative $L_\\infty$ Error")
    fig.suptitle(title, y=1.02, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_radial_error_profiles(results_list, title="Radial Ray Error Profiles"):
    """
    Plot pointwise error |u_approx - u_true| along r at a fixed theta ray.
    """
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=150)

    sorted_results = sorted(
        results_list,
        key=lambda r: (
            0 if ("uniform" in str(r["rad_kind"]).lower() and "non" not in str(r["rad_kind"]).lower())
            else (1 if "chebyshev" in str(r["rad_kind"]).lower() else 2)
        )
    )
    for res in sorted_results:
        label = normalize_radial_label(res["rad_kind"])
        r_m = res["r_m"]
        err_ray = np.abs(res["u_approx"][0, :] - res["u_true"][0, :])
        ax.semilogy(r_m, err_ray + 1e-16, label=f"{label} (M={res['M']})", linewidth=1.2)

    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Radial Distance $r$")
    ax.set_ylabel("Pointwise Error $|u_{approx}(r) - u_{true}(r)|$")
    ax.set_title(title)
    ax.legend(frameon=True, facecolor="white", edgecolor="0.8")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# HTML Table Rendering Routines (Separate Accuracy & Runtime Tables)
# ==============================================================================

def render_nxm_accuracy_table(df_results, value_col="Linf_rel", title="Accuracy Table: Relative Linf Error"):
    """
    Render N x M Accuracy table (Pivot: index N, columns [rad_kind, M]).
    Accuracies rendered FIRST.
    Column ordering: Uniform Radial -> Chebyshev-Lobatto -> Nonuniform.
    """
    if title:
        display(HTML(f"<h4 style='font-size: 13px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>{title}</h4>"))
    df = df_results.copy()
    df["rad_kind"] = df["rad_kind"].apply(normalize_radial_label)
    pivot = df.pivot_table(index="N", columns=["rad_kind", "M"], values=value_col, aggfunc="first")
    ordered_rad_kinds = sort_radial_columns(pivot.columns.levels[0])
    pivot = pivot.reindex(columns=ordered_rad_kinds, level=0)
    display(HTML(pivot.map(lambda v: f"{v:.2e}" if np.isfinite(v) else "—").to_html(classes="table table-bordered text-center")))


def render_nxm_runtime_table(df_results, title="Runtime Table: Solve Time (ms)"):
    """
    Render N x M Runtime table (Pivot: index N, columns [rad_kind, M]).
    Runtimes rendered LAST.
    Column ordering: Uniform Radial -> Chebyshev-Lobatto -> Nonuniform.
    """
    if title:
        display(HTML(f"<h4 style='font-size: 13px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>{title}</h4>"))
    df = df_results.copy()
    df["rad_kind"] = df["rad_kind"].apply(normalize_radial_label)
    pivot = df.pivot_table(index="N", columns=["rad_kind", "M"], values="runtime", aggfunc="first") * 1000.0
    ordered_rad_kinds = sort_radial_columns(pivot.columns.levels[0])
    pivot = pivot.reindex(columns=ordered_rad_kinds, level=0)
    display(HTML(pivot.map(lambda v: f"{v:.2f} ms" if np.isfinite(v) else "—").to_html(classes="table table-bordered text-center")))


def render_fixed_n_accuracy_table(df_results, title="Accuracy Table (Fixed N, Varying M)"):
    """
    Render Fixed-N Accuracy table (Relative L2 and Linf errors). Accuracies FIRST.
    Column ordering: Uniform Radial -> Chebyshev-Lobatto -> Nonuniform.
    """
    if title:
        display(HTML(f"<h4 style='font-size: 13px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>{title}</h4>"))
    df = df_results.copy()
    df["rad_kind"] = df["rad_kind"].apply(normalize_radial_label)

    pivot_linf = df.pivot_table(index="M", columns="rad_kind", values="Linf_rel", aggfunc="first")
    pivot_l2 = df.pivot_table(index="M", columns="rad_kind", values="L2_rel", aggfunc="first")

    ordered_cols = sort_radial_columns(pivot_linf.columns)
    pivot_linf = pivot_linf.reindex(columns=ordered_cols)
    pivot_l2 = pivot_l2.reindex(columns=ordered_cols)

    combined = pd.concat([pivot_linf], keys=["Relative Linf Error"], axis=1)
    combined_l2 = pd.concat([pivot_l2], keys=["Relative L2 Error"], axis=1)
    full_acc = pd.concat([combined, combined_l2], axis=1)

    display(HTML(full_acc.map(lambda v: f"{v:.2e}" if np.isfinite(v) else "—").to_html(classes="table table-bordered text-center")))


def render_fixed_n_runtime_table(df_results, title="Runtime Table (Fixed N, Varying M)"):
    """
    Render Fixed-N Runtime table (Solve time in ms). Runtimes LAST.
    Column ordering: Uniform Radial -> Chebyshev-Lobatto -> Nonuniform.
    """
    if title:
        display(HTML(f"<h4 style='font-size: 13px; font-weight: bold; margin-top: 15px; margin-bottom: 5px;'>{title}</h4>"))
    df = df_results.copy()
    df["rad_kind"] = df["rad_kind"].apply(normalize_radial_label)

    pivot = df.pivot_table(index="M", columns="rad_kind", values="runtime", aggfunc="first") * 1000.0
    ordered_cols = sort_radial_columns(pivot.columns)
    pivot = pivot.reindex(columns=ordered_cols)

    display(HTML(pivot.map(lambda v: f"{v:.2f} ms" if np.isfinite(v) else "—").to_html(classes="table table-bordered text-center")))

