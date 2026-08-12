import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, HTML
from tqdm.auto import tqdm

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_cartesian_grid_on_disk,
    generate_grid_values,
    compute_zero_mode,
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver

import sympy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------
# Exact Analytical Benchmark with Localized Azimuthal Peak
# ---------------------------------------------------------
def get_azimuthal_gaussian_ridge_problem(R=1.0, theta_0=np.pi, sigma=0.12):
    """
    Creates a problem with a localized Gaussian ridge.
    Analytically pre-simplified to prevent floating point blowups at r=0.
    """
    r_sym, th_sym = sp.symbols('r th', real=True)
    
    # Angular peak function
    angular_peak = sp.exp(- (sp.sin((th_sym - theta_0) / 2.0) / sigma)**2)
    
    # Pre-calculate the exact angular derivatives
    peak_func = sp.lambdify(th_sym, angular_peak, "numpy")
    peak_thth_func = sp.lambdify(th_sym, sp.diff(angular_peak, th_sym, 2), "numpy")

    def u_true(xc, yc):
        r = np.sqrt(xc**2 + yc**2)
        th = np.arctan2(yc, xc)
        return r**2 * (R**2 - r**2) * peak_func(th)

    def f_rhs(xc, yc):
        r = np.sqrt(xc**2 + yc**2)
        th = np.arctan2(yc, xc)
        
        # ANALYTICALLY SIMPLIFIED LAPLACIAN (No division by r!)
        # Delta u = u_rr + u_r/r + u_thth/r^2
        # Since u = r^2(R^2 - r^2)*P(th), the 1/r^2 cancels perfectly.
        radial_part = (4.0 * R**2 - 16.0 * r**2) * peak_func(th)
        angular_part = (R**2 - r**2) * peak_thth_func(th)
        
        return radial_part + angular_part

    def g_dirichlet(xc, yc):
        return np.zeros_like(xc)

    def g_neumann(xc, yc, R_val=R):
        thc = np.arctan2(yc, xc)
        return -2.0 * R_val**3 * peak_func(thc)

    return {
        "u": u_true,
        "f": f_rhs,
        "g_dirichlet": g_dirichlet,
        "g_neumann": g_neumann,
        "R": R,
        "sigma": sigma,
        "theta_0": theta_0
    }
# ---------------------------------------------------------
# Smooth Adapted Clustered Azimuthal Grid Generator
# ---------------------------------------------------------
def generate_adapted_clustered_azimuthal(N, cluster_strength=0.40, center=np.pi):
    """
    Generates N non-uniform angles in [0, 2π) smoothly clustered around 'center' (e.g. θ_0).

    Uses the smooth deformation mapping:
        theta(s) = s - cluster_strength * sin(s - center)

    This ensures:
    - Bijective, strictly monotonic angle mapping (for 0 <= cluster_strength < 1).
    - Dense angular nodes where the gradient is sharp (theta ~ center).
    - Preserved frame conditioning (no unbounded gaps).
    """
    if not (0.0 <= cluster_strength < 1.0):
        raise ValueError("cluster_strength must satisfy 0 <= cluster_strength < 1.0")

    s = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    theta = s - cluster_strength * np.sin(s - center)
    theta = np.mod(theta, 2.0 * np.pi)
    return np.sort(theta)

# ---------------------------------------------------------
# Periodic 1-D linear interpolation helper
# ---------------------------------------------------------
def periodic_linear_interpolate(theta_src, values_src, theta_tgt):
    """
    Periodically interpolates `values_src` sampled at sorted `theta_src`
    (length P) onto `theta_tgt` (length Q).

    • theta_src, theta_tgt are 1-D arrays in [0, 2π).
    • values_src can be shape (P,) or (P, K).
    """
    theta_src  = np.asarray(theta_src,  dtype=float)
    values_src = np.asarray(values_src)

    # Build periodic extension (prepend and append one point)
    theta_ext  = np.concatenate(
        ([theta_src[-1] - 2.0*np.pi], theta_src, [theta_src[0] + 2.0*np.pi])
    )
    if values_src.ndim == 1:
        values_ext = np.concatenate(([values_src[-1]], values_src, [values_src[0]]))
        return np.interp(theta_tgt, theta_ext, values_ext)

    # 2-D (P × K): extend along first axis
    values_ext = np.concatenate(
        (values_src[-1:, :], values_src, values_src[:1, :]),
        axis=0
    )
    return np.vstack([
        np.interp(theta_tgt, theta_ext, values_ext[:, k])
        for k in range(values_ext.shape[1])
    ]).T          # return shape (Q, K)


# ======================================================================
# 1)  run_benchmark_case  – correct grid handling, timing, interpolation
# ======================================================================
def run_benchmark_case(
        N, M,
        azu_unif,            # 1 = NUFFT / NUDFT   | 2 = Uniform FFT (+interp)
        theta_j,
        problem,
        bc_choice=1, quad_rule=2,
        use_nudft=False,
        maxiter_nufft=100, tol_nufft=1e-10,
        cluster_strength=0.40, theta_0=np.pi):
    """
    Solve the Poisson problem on a disk.

    • Uniform radial grid of size M.
    • `theta_j` is the clustered angular grid where the data are *measured*.
    • If `azu_unif == 2`, the routine interpolates those measurements to a
      uniform θ grid before calling the Uniform-FFT solver.
    """

    # ---------- unpack benchmark problem ----------
    R         = problem["R"]
    u_true_f  = problem["u"]
    f_rhs_f   = problem["f"]
    g_dir_f   = problem["g_dirichlet"]
    g_neu_f   = problem["g_neumann"]

    # ---------- grids ----------
    r_m       = generate_uniform_radial(M, R)              # always uniform
    theta_raw = np.asarray(theta_j, dtype=float)           # measurement angles

    # Cartesian coords on measurement grid (for sampling RHS)
    x_raw, y_raw = generate_cartesian_grid_on_disk(theta_raw, r_m)

    # ---------- acquire scattered data ----------
    f_raw = f_rhs_f(x_raw, y_raw)

    if bc_choice == 1:
        g_raw = g_dir_f(x_raw[:, -1], y_raw[:, -1])
    else:
        g_raw = g_neu_f(x_raw[:, -1], y_raw[:, -1], R)

    # ========= start timing *before* any interpolation =========
    t0 = time.perf_counter()

    # ---------- uniform FFT path needs gridding / interpolation ----------
    if azu_unif == 2:
        theta_uniform = generate_uniform_azimuthal(N)

        # Interpolate all radial rings at once.
        f_values = periodic_linear_interpolate(
            theta_raw,
            f_raw,
            theta_uniform,
        )

        g_values = periodic_linear_interpolate(
            theta_raw,
            g_raw,
            theta_uniform,
        )
        theta_solver = theta_uniform
    else:
        # NUFFT / NUDFT: consume data directly
        f_values     = f_raw
        g_values     = g_raw
        theta_solver = theta_raw

    # ---------- exact solution on *solver* grid (for error) ----------
    x_sol, y_sol = generate_cartesian_grid_on_disk(theta_solver, r_m)
    u_ref_solver = u_true_f(x_sol, y_sol)

    # ---------- zero-mode constraint ----------
    if bc_choice == 2 or azu_unif == 1:
        u_fourier_0_arr = compute_zero_mode(
            u_ref_solver, theta_solver, azu_unif
        )
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    # ---------- solve ----------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_approx = poisson_solver(
            f_values=f_values,
            g_values=g_values,
            u_fourier_0=u_fourier_0,
            N=N, M=M,
            r_m=r_m,
            theta_j=theta_solver,
            R=R,
            quad_rule=quad_rule,
            BC_choice=bc_choice,
            rad_unif=1,
            azu_unif=azu_unif,
            grid_type=(1 if azu_unif == 2 else 3),
            use_nudft_angular=use_nudft,
            maxiter_nufft=maxiter_nufft,
            tol_nufft=tol_nufft,
        )

    runtime = time.perf_counter() - t0

    # ---------- error ----------
    _, linf_rel, _, l2_rel = compute_error_metrics(
        u_approx, u_ref_solver, r_m, theta_solver
    )

    return {
        "N": N, "M": M,
        "azu_unif": azu_unif,
        "use_nudft": use_nudft,
        "quad_rule": quad_rule,
        "bc_choice": bc_choice,
        "L2_rel": l2_rel,
        "Linf_rel": linf_rel,
        "runtime": runtime,
        "u_approx": u_approx,
        "u_true":   u_ref_solver,
        "x_coord":  x_sol,
        "y_coord":  y_sol,
        "iRadius":  r_m,
        "iAngle":   theta_solver,
    }

# ======================================================================
# 2)  run_all_algorithms_NxM_study  (all methods share same clustered θ)
# ======================================================================
def run_all_algorithms_NxM_study(problem, N_values, M_values,
                                 cluster_strength=0.40, theta_0=np.pi,
                                 bc_choice=1, quad_rule=2):
    """
    Computes the (N × M) grid for the three angular solvers:
      1. Adapted NUFFT
      2. Adapted NUDFT
      3. Uniform FFT  (uses interpolation step internally)
    All methods start from the *same* clustered θ grid so that they
    receive identical data samples.
    """
    algorithms = [
        ("Adapted NUFFT",                     1, False),
        ("Adapted NUDFT",                     1, True),
        ("Uniform FFT + linear interpolation", 2, False),
    ]

    results = []
    total   = len(algorithms) * len(N_values) * len(M_values)
    pbar = tqdm(total=total, desc="Computing N×M matrix", leave=True)

    for method_name, azu_unif, use_nudft in algorithms:
        for N in N_values:
            # Build *one* deterministic clustered θ grid for this N
            theta_clust = generate_adapted_clustered_azimuthal(
                N, cluster_strength=cluster_strength, center=theta_0
            )

            for M in M_values:
                res = run_benchmark_case(
                    N=N, M=M,
                    azu_unif=azu_unif,
                    theta_j=theta_clust,   # always pass clustered grid
                    problem=problem,
                    bc_choice=bc_choice,
                    quad_rule=quad_rule,
                    use_nudft=use_nudft,
                    cluster_strength=cluster_strength,
                    theta_0=theta_0,
                )
                res["method"] = method_name
                results.append(res)
                pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


# ---------------------------------------------------------
# Visualization Utilities
# ---------------------------------------------------------
def plot_solution_and_grids(problem, N_adapt=32, N_unif=32, M=32,
                            cluster_strength=0.40, theta_0=np.pi):
    """
    Clean visual layout:
    1. 3D surface plot of exact analytical solution on disk.
    2. 2D polar lattice scatter comparison (clean guide rings + boundary nodes).
    3. 1D angular profile showing node density over wave peak.
    """
    R = problem["R"]
    theta_adapt = generate_adapted_clustered_azimuthal(N_adapt, cluster_strength=cluster_strength, center=theta_0)
    theta_unif = generate_uniform_azimuthal(N_unif)
    radii = generate_uniform_radial(M, R)

    th_fine = np.linspace(0, 2 * np.pi, 250)
    r_fine = np.linspace(0, R, 120)
    Xf, Yf = generate_cartesian_grid_on_disk(th_fine, r_fine)
    Zf = problem["u"](Xf, Yf)

    fig = plt.figure(figsize=(11, 3.5))

    # Subplot 1: 3D True Solution Surface
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(Xf, Yf, Zf, cmap='plasma', edgecolor='none', alpha=0.95)
    ax1.set_title(r"Exact Solution $u(x,y)$ on Disk", fontsize=12, pad=10)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # Subplot 2: Clean 2D Polar Grid Comparison
    ax2 = fig.add_subplot(1, 3, 2)
    # Draw only 5 guide rings for visual clarity
    for r_guide in np.linspace(0.2 * R, R, 5):
        ax2.add_patch(Circle((0, 0), r_guide, fill=False, ec="0.82", lw=0.8, linestyle="--"))
    ax2.add_patch(Circle((0, 0), R, fill=False, ec="black", lw=1.5))

    # Plot sample nodes on 3 representative rings (0.4R, 0.7R, R)
    sample_r = [0.4 * R, 0.7 * R, R]
    for r_s in sample_r:
        x_un = r_s * np.cos(theta_unif)
        y_un = r_s * np.sin(theta_unif)
        x_ad = r_s * np.cos(theta_adapt)
        y_ad = r_s * np.sin(theta_adapt)
        ax2.scatter(x_un, y_un, s=16, color='royalblue', alpha=0.6,
                    label=f"Uniform FFT target grid (N={N_unif})" if r_s == R else "")
        ax2.scatter(x_ad, y_ad, s=24, color='crimson', marker='^', alpha=0.85,
                    label=f"Scattered measurement grid (N={N_adapt})" if r_s == R else "")

    ax2.set_title("Polar Grid Comparison on Disk", fontsize=12)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.15 * R, 1.15 * R)
    ax2.set_ylim(-1.15 * R, 1.15 * R)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.axis("off")

    # Subplot 3: 1D Angular Profile & Node Placement
    ax3 = fig.add_subplot(1, 3, 3)
    r_eval = 0.70 * R
    u_slice = [problem["u"](r_eval * np.cos(t), r_eval * np.sin(t)) for t in th_fine]
    ax3.plot(th_fine, u_slice, 'k-', lw=2, label=rf"$u(r={r_eval:.1f}R, \theta)$ Profile")

    u_nodes_ad = [problem["u"](r_eval * np.cos(t), r_eval * np.sin(t)) for t in theta_adapt]
    u_nodes_un = [problem["u"](r_eval * np.cos(t), r_eval * np.sin(t)) for t in theta_unif]

    ax3.scatter(theta_unif, u_nodes_un, color='royalblue', s=35, zorder=3, label=f"Uniform Nodes (N={N_unif})")
    ax3.scatter(theta_adapt, u_nodes_ad, color='crimson', s=45, marker='^', zorder=4, label=f"Adapted Nodes (N={N_adapt})")
    ax3.set_xlabel(r"Azimuthal Angle $\theta$ (rad)", fontsize=8)
    ax3.set_ylabel(r"$u(0.7R, \theta)$", fontsize=8)
    ax3.set_title("Node Density over Sharp Angular Wave", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.show()



def plot_adapted_vs_highres_uniform(problem, N_adapt=32, N_unif_high=256, M=64,
                                    cluster_strength=0.40, theta_0=np.pi,
                                    quad_rule=2, bc_choice=1):
    """
    Renders side-by-side 3D error surfaces comparing:
    1. Adapted NUFFT (low N = 32)
    2. Adapted NUDFT (low N = 32)
    3. Uniform FFT (cranked up high-resolution N = 256 or 512)
    """
    fig = plt.figure(figsize=(11, 3.5))

    # Case 1: Adapted NUFFT
    th_ad = generate_adapted_clustered_azimuthal(N_adapt, cluster_strength=cluster_strength, center=theta_0)
    res_nufft = run_benchmark_case(
        N=N_adapt, M=M, azu_unif=1, theta_j=th_ad, problem=problem,
        bc_choice=bc_choice, quad_rule=quad_rule, use_nudft=False
    )

    # Case 2: Adapted NUDFT
    res_nudft = run_benchmark_case(
        N=N_adapt, M=M, azu_unif=1, theta_j=th_ad, problem=problem,
        bc_choice=bc_choice, quad_rule=quad_rule, use_nudft=True
    )

    # Case 3: High-resolution Uniform FFT
    # Case 3: High-resolution Uniform FFT
    th_un_high = generate_adapted_clustered_azimuthal(
        N_unif_high, cluster_strength=cluster_strength, center=theta_0
    )
    res_unif = run_benchmark_case(
        N=N_unif_high, M=M, azu_unif=2, theta_j=th_un_high, problem=problem,
        bc_choice=bc_choice, quad_rule=quad_rule, use_nudft=False
    )

    cases = [
    (
        f"Adapted NUFFT — direct data\n"
        f"(N={N_adapt}, M={M})",
        res_nufft,
    ),
    (
        f"Adapted NUDFT — direct data\n"
        f"(N={N_adapt}, M={M})",
        res_nudft,
    ),
    (
        f"Uniform FFT + linear interpolation\n"
        f"(N={N_unif_high} measurements, M={M})",
        res_unif,
    ),
    ]

    for i, (title, res) in enumerate(cases, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        err = np.abs(res["u_true"] - res["u_approx"])
        X = res["x_coord"]
        Y = res["y_coord"]

        Xp = np.vstack([X, X[0, :]])
        Yp = np.vstack([Y, Y[0, :]])
        Ep = np.vstack([err, err[0, :]])

        surf = ax.plot_surface(Xp, Yp, Ep, cmap='inferno', edgecolor='none')
        ax.set_title(f"{title}\n$L_2$ Error = {res['L2_rel']:.2e} | t = {res['runtime']:.4f}s", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Pointwise Error")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.show()

def render_combined_runtime_table(df_results):
    """
    Render a single table with:
      - Rows: N (angular points)
      - Columns: (method, M) as a MultiIndex

    So each method forms a multi-column block over M.
    """
    # Build a pivot: index N, columns (method, M), values runtime
    pivot = df_results.pivot_table(
        index="N",
        columns=["method", "M"],
        values="runtime"
    )

    # Sort columns by method then M for readability
    pivot = pivot.sort_index(axis=1)

    # Format as readable strings
    formatted = pivot.copy()
    formatted = formatted.map(lambda v: f"{v:.4f}" if np.isfinite(v) else "—")

    print(f"\n{'='*75}\nCombined Runtime Table — Methods × M (index N)\n{'='*75}")
    display(HTML(formatted.to_html(classes="table table-bordered table-striped text-center")))

    return pivot

def render_combined_error_table(df_results, value_col="L2_rel"):
    """
    Render one N-vs-M error table for every method.

    Rows:
        N (angular grid points)

    Column groups:
        Method -> M (radial grid points)

    Example:
                         Adapted NUFFT              Adapted NUDFT              Uniform FFT
                    M=16   M=32  M=64 ...     M=16   M=32  M=64 ...     M=16   M=32  M=64 ...
        N=16         ...    ...    ...          ...    ...    ...          ...    ...    ...
        N=32         ...    ...    ...          ...    ...    ...          ...    ...    ...
    """
    method_order = [
        "Adapted NUFFT",
        "Adapted NUDFT",
        "Uniform FFT + linear interpolation",
    ]

    # One row per N; hierarchical columns: method -> M.
    pivot = df_results.pivot_table(
        index="N",
        columns=["method", "M"],
        values=value_col,
        aggfunc="first",
    )

    # Preserve the intended method order and sort M within each method.
    available_methods = [
        method for method in method_order
        if method in df_results["method"].unique()
    ]

    ordered_columns = [
        (method, M)
        for method in available_methods
        for M in sorted(df_results["M"].unique())
        if (method, M) in pivot.columns
    ]

    pivot = pivot.reindex(columns=ordered_columns).sort_index()

    formatted = pivot.copy()
    formatted = formatted.map(
        lambda value: f"{value:.2e}" if np.isfinite(value) else "—"
    )

    if value_col == "L2_rel":
        title = "Relative L2 Error"
    elif value_col == "Linf_rel":
        title = "Relative L∞ Error"
    else:
        title = value_col

    print(f"\n{'=' * 90}\n{title} — N vs M, All Methods\n{'=' * 90}")

    display(
        HTML(
            formatted.to_html(
                classes="table table-bordered table-striped text-center",
                border=0,
            )
        )
    )

    return pivot

def plot_extreme_runtime_2x2(df_results, N_min=None, N_max=None, M_min=None, M_max=None):
    """
    Plot a 2×2 panel of runtimes for:
      - Top-left: N = N_min, runtime vs M
      - Top-right: N = N_max, runtime vs M
      - Bottom-left: M = M_min, runtime vs N
      - Bottom-right: M = M_max, runtime vs N

    All panels share the same y-axis limits.
    """
    # Infer extremes if not provided
    if N_min is None:
        N_min = df_results["N"].min()
    if N_max is None:
        N_max = df_results["N"].max()
    if M_min is None:
        M_min = df_results["M"].min()
    if M_max is None:
        M_max = df_results["M"].max()

    methods = sorted(df_results["method"].unique())

    colors = {
        "Adapted NUFFT": "crimson",
        "Adapted NUDFT": "forestgreen",
        "Uniform FFT + linear interpolation": "royalblue",
    }
    markers = {
        "Adapted NUFFT": "s-",
        "Adapted NUDFT": "^-",
        "Uniform FFT + linear interpolation": "o-",
    }

    # Global runtime range for shared y-axis
    rmin = df_results["runtime"].min()
    rmax = df_results["runtime"].max()
    # avoid zero issues on log scale
    if rmin <= 0:
        rmin = df_results[df_results["runtime"] > 0]["runtime"].min()
    rmin *= 0.8
    rmax *= 1.2

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8))
    (ax11, ax12), (ax21, ax22) = axes

    # Top-left: N = N_min, runtime vs M
    sub_Nmin = df_results[df_results["N"] == N_min]
    for meth, grp in sub_Nmin.groupby("method"):
        grp_s = grp.sort_values("M")
        ax11.loglog(
            grp_s["M"], grp_s["runtime"],
            markers.get(meth, "o-"),
            color=colors.get(meth, "black"),
            lw=1.8, ms=5, label=meth,
        )
    ax11.set_xlabel("Radial Grid Points $M$")
    ax11.set_ylabel("Runtime (seconds)")
    ax11.set_title(f"$N = {N_min}$ — Runtime vs $M$")
    ax11.grid(True, which="both", ls="--", alpha=0.5)
    ax11.set_ylim(rmin, rmax)

    # Top-right: N = N_max, runtime vs M
    sub_Nmax = df_results[df_results["N"] == N_max]
    for meth, grp in sub_Nmax.groupby("method"):
        grp_s = grp.sort_values("M")
        ax12.loglog(
            grp_s["M"], grp_s["runtime"],
            markers.get(meth, "o-"),
            color=colors.get(meth, "black"),
            lw=1.8, ms=5, label=meth,
        )
    ax12.set_xlabel("Radial Grid Points $M$")
    ax12.set_ylabel("Runtime (seconds)")
    ax12.set_title(f"$N = {N_max}$ — Runtime vs $M$")
    ax12.grid(True, which="both", ls="--", alpha=0.5)
    ax12.set_ylim(rmin, rmax)

    # Bottom-left: M = M_min, runtime vs N
    sub_Mmin = df_results[df_results["M"] == M_min]
    for meth, grp in sub_Mmin.groupby("method"):
        grp_s = grp.sort_values("N")
        ax21.loglog(
            grp_s["N"], grp_s["runtime"],
            markers.get(meth, "o-"),
            color=colors.get(meth, "black"),
            lw=1.8, ms=5, label=meth,
        )
    ax21.set_xlabel("Angular Grid Points $N$")
    ax21.set_ylabel("Runtime (seconds)")
    ax21.set_title(f"$M = {M_min}$ — Runtime vs $N$")
    ax21.grid(True, which="both", ls="--", alpha=0.5)
    ax21.set_ylim(rmin, rmax)

    # Bottom-right: M = M_max, runtime vs N
    sub_Mmax = df_results[df_results["M"] == M_max]
    for meth, grp in sub_Mmax.groupby("method"):
        grp_s = grp.sort_values("N")
        ax22.loglog(
            grp_s["N"], grp_s["runtime"],
            markers.get(meth, "o-"),
            color=colors.get(meth, "black"),
            lw=1.8, ms=5, label=meth,
        )
    ax22.set_xlabel("Angular Grid Points $N$")
    ax22.set_ylabel("Runtime (seconds)")
    ax22.set_title(f"$M = {M_max}$ — Runtime vs $N$")
    ax22.grid(True, which="both", ls="--", alpha=0.5)
    ax22.set_ylim(rmin, rmax)

    # One shared legend
    handles, labels = ax11.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), fontsize=9, frameon=False)

    fig.suptitle("Chebyshev Problem 1 — Extreme N,M Runtime Comparisons", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_accuracy_from_df(df_results, fixed_M=64, fixed_N=64):
    """
    Two-panel accuracy comparison:
      Left:  L2 error vs N at fixed M.
      Right: L2 error vs M at fixed N.

    Both panels share the same y-axis.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7.5, 3.2),
        sharey=True
    )

    colors = {
        "Adapted NUFFT": "crimson",
        "Adapted NUDFT": "forestgreen",
        "Uniform FFT + linear interpolation": "royalblue",
    }
    markers = {
        "Adapted NUFFT": "s--",
        "Adapted NUDFT": "^-.",
        "Uniform FFT + linear interpolation": "o-",
    }

    # Left: error vs N at fixed M
    sub_M = df_results[df_results["M"] == fixed_M]
    for meth, grp in sub_M.groupby("method"):
        grp_s = grp.sort_values("N")
        ax1.loglog(
            grp_s["N"], grp_s["L2_rel"],
            markers.get(meth, "o-"),
            color=colors.get(meth, "black"),
            lw=1.5, ms=4, label=meth,
        )

    ax1.set_xlabel("Angular points $N$", fontsize=9)
    ax1.set_ylabel(r"Relative $L_2$ error", fontsize=9)
    ax1.set_title(f"Error vs $N$ ($M={fixed_M}$)", fontsize=8)
    ax1.grid(True, which="both", linestyle="--", alpha=0.45)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=7)

    # Right: error vs M at fixed N
    sub_N = df_results[df_results["N"] == fixed_N]
    for meth, grp in sub_N.groupby("method"):
        grp_s = grp.sort_values("M")
        ax2.loglog(
            grp_s["M"], grp_s["L2_rel"],
            markers.get(meth, "o-"),
            color=colors.get(meth, "black"),
            lw=1.5, ms=4, label=meth,
        )

    ax2.set_xlabel("Radial points $M$", fontsize=9)
    ax2.set_title(f"Error vs $M$ ($N={fixed_N}$)", fontsize=8)
    ax2.grid(True, which="both", linestyle="--", alpha=0.45)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7)

    fig.suptitle("Chebyshev Problem 1 — Accuracy Comparison", fontsize=8, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_accuracy_faceted_by_method(df_results, N_values=None, M_values=None):
    """
    2 x K accuracy plot:
      Top row: Error vs M, one curve per N; shared y-axis across methods.
      Bottom row: Error vs N, one curve per M; shared y-axis across methods.
    """
    methods = sorted(df_results["method"].unique())
    K = len(methods)

    if N_values is None:
        N_values = sorted(df_results["N"].unique())
    if M_values is None:
        M_values = sorted(df_results["M"].unique())

    fig, axes = plt.subplots(
        2, K,
        figsize=(3.5 * K, 5.0),
        sharey="row",
        squeeze=False
    )

    cmap = plt.get_cmap("tab10")

    for col, meth in enumerate(methods):
        sub = df_results[df_results["method"] == meth]

        # Top row: Error vs M, one curve for each N
        ax_top = axes[0, col]
        for i, N in enumerate(N_values):
            grp = sub[sub["N"] == N].sort_values("M")
            if grp.empty:
                continue

            ax_top.loglog(
                grp["M"], grp["L2_rel"],
                marker="o", ms=3.5, lw=1.2,
                color=cmap(i % 10),
                label=f"N={N}",
            )

        ax_top.set_title(f"{meth}\nError vs $M$", fontsize=9)
        ax_top.set_xlabel("Radial points $M$", fontsize=8)
        ax_top.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_top.tick_params(labelsize=7)
        ax_top.legend(fontsize=6, loc="best")

        if col == 0:
            ax_top.set_ylabel(r"Relative $L_2$ error", fontsize=8)

        # Bottom row: Error vs N, one curve for each M
        ax_bot = axes[1, col]
        for i, M in enumerate(M_values):
            grp = sub[sub["M"] == M].sort_values("N")
            if grp.empty:
                continue

            ax_bot.loglog(
                grp["N"], grp["L2_rel"],
                marker="s", ms=3.5, lw=1.2,
                color=cmap(i % 10),
                label=f"M={M}",
            )

        ax_bot.set_title(f"{meth}\nError vs $N$", fontsize=9)
        ax_bot.set_xlabel("Angular points $N$", fontsize=8)
        ax_bot.grid(True, which="both", linestyle="--", alpha=0.45)
        ax_bot.tick_params(labelsize=7)
        ax_bot.legend(fontsize=6, loc="best")

        if col == 0:
            ax_bot.set_ylabel(r"Relative $L_2$ error", fontsize=8)

    fig.suptitle("Chebyshev Problem 1 — Accuracy by Method", fontsize=8, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

