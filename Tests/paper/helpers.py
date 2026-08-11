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


# ---------------------------------------------------------
# Exact Analytical Benchmark with Localized Azimuthal Peak
# ---------------------------------------------------------
def get_azimuthal_benchmark_problem(R=1.0, epsilon=0.05):
    """
    Chebyshev Problem 1: Angular boundary layer on the disk B(0,R).

        u(r, theta) = r^2 * (R^2 - r^2) * H(theta),
        H(theta) = tanh(theta / epsilon),  epsilon << 1

    Properties:
    - Homogeneous Dirichlet boundary condition: u(R, theta) = 0.
    - Smooth at origin: u(0, theta) = 0.
    - Thin angular boundary layer of width O(epsilon) near theta = 0.
    - Exact Laplacian (same radial factor as the wave-packet case):

        Δu = (4*R^2 - 16*r^2) * H(theta) + (R^2 - r^2) * H''(theta)

      where H'' is the second derivative of tanh(theta/epsilon).
    """
    th_sym = sp.symbols('th', real=True)

    # Angular profile and its second derivative
    H_sym = sp.tanh(th_sym / epsilon)
    H_d2_sym = sp.diff(H_sym, th_sym, 2)

    H_func = sp.lambdify(th_sym, H_sym, "numpy")
    H_d2_func = sp.lambdify(th_sym, H_d2_sym, "numpy")

    def u_true(xc, yc):
        rc = np.sqrt(xc**2 + yc**2)
        thc = np.arctan2(yc, xc)
        return rc**2 * (R**2 - rc**2) * H_func(thc)

    def f_rhs(xc, yc):
        rc = np.sqrt(xc**2 + yc**2)
        thc = np.arctan2(yc, xc)
        # Radial part (same derivation as in your original code)
        return (4.0 * R**2 - 16.0 * rc**2) * H_func(thc) + (R**2 - rc**2) * H_d2_func(thc)

    def g_dirichlet(xc, yc):
        # For Dirichlet, we just use the exact solution on the boundary.
        return u_true(xc, yc)

    def g_neumann(xc, yc, R_val=R):
        # Radial derivative of u at r = R:
        # u(r,theta) = r^2(R^2-r^2)H(theta) => ∂u/∂r|_{r=R} = -2 R^3 H(theta)
        rc = np.sqrt(xc**2 + yc**2)
        thc = np.arctan2(yc, xc)
        return -2.0 * (R_val**3) * H_func(thc)

    return {
        "u": u_true,
        "f": f_rhs,
        "g_dirichlet": g_dirichlet,
        "g_neumann": g_neumann,
        "H_func": H_func,
        "R": R,
        "epsilon": epsilon,
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
# Single Case Solver Harness
# ---------------------------------------------------------
def run_benchmark_case(N, M, azu_unif, theta_j, problem,
                       bc_choice=1, quad_rule=2, use_nudft=False,
                       maxiter_nufft=100, tol_nufft=1e-10):
    """
    Solves the 2D Poisson equation on the disk using a uniform radial grid
    and the specified azimuthal grid (uniform or non-uniform).
    """
    R = problem["R"]
    u_true = problem["u"]
    f_rhs = problem["f"]
    g_dir = problem["g_dirichlet"]
    g_neu = problem["g_neumann"]

    iRadius = generate_uniform_radial(M, R)
    iAngle = np.asarray(theta_j, dtype=float)

    x_coord, y_coord = generate_cartesian_grid_on_disk(iAngle, iRadius)
    f_values = f_rhs(x_coord, y_coord)
    u_t = u_true(x_coord, y_coord)

    if bc_choice == 1:
        g_values = g_dir(x_coord[:, M - 1], y_coord[:, M - 1])
    else:
        g_values = g_neu(x_coord[:, M - 1], y_coord[:, M - 1], R)

    if bc_choice == 2:
        u_fourier_0_arr = compute_zero_mode(u_t, iAngle, azu_unif)
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_approx = poisson_solver(
            f_values=f_values,
            g_values=g_values,
            u_fourier_0=u_fourier_0,
            N=N,
            M=M,
            r_m=iRadius,
            theta_j=iAngle,
            R=R,
            quad_rule=quad_rule,
            BC_choice=bc_choice,
            rad_unif=1,               # radial grid is always uniform here
            azu_unif=azu_unif,
            grid_type=(1 if azu_unif == 2 else 3),
            use_nudft_angular=use_nudft,
            maxiter_nufft=maxiter_nufft,
            tol_nufft=tol_nufft,
        )
    runtime = time.perf_counter() - t0

    _, linf_rel, _, l2_rel = compute_error_metrics(u_approx, u_t, iRadius, iAngle)

    return {
        "N": N,
        "M": M,
        "azu_unif": azu_unif,
        "use_nudft": use_nudft,
        "quad_rule": quad_rule,
        "bc_choice": bc_choice,
        "L2_rel": l2_rel,
        "Linf_rel": linf_rel,
        "runtime": runtime,
        "u_approx": u_approx,
        "u_true": u_t,
        "x_coord": x_coord,
        "y_coord": y_coord,
        "iRadius": iRadius,
        "iAngle": iAngle
    }


# ---------------------------------------------------------
# N vs. M Grid Study for All 3 Algorithms
# ---------------------------------------------------------
def run_all_algorithms_NxM_study(problem, N_values, M_values,
                                 cluster_strength=0.40, theta_0=np.pi,
                                 bc_choice=1, quad_rule=2):
    """
    Computes the full (N x M) grid evaluations for:
    1. Adapted NUFFT
    2. Adapted NUDFT
    3. Uniform FFT
    """
    algorithms = [
        ("Adapted NUFFT", 1, False),
        ("Adapted NUDFT", 1, True),
        ("Uniform FFT", 2, False)
    ]

    results = []
    total_runs = len(algorithms) * len(N_values) * len(M_values)
    pbar = tqdm(total=total_runs, desc="Computing N vs M Grid Matrix", leave=True)

    for method_name, azu_unif, use_nudft in algorithms:
        for N in N_values:
            if azu_unif == 1:
                theta = generate_adapted_clustered_azimuthal(N, cluster_strength=cluster_strength, center=theta_0)
            else:
                theta = generate_uniform_azimuthal(N)

            for M in M_values:
                res = run_benchmark_case(
                    N=N, M=M, azu_unif=azu_unif, theta_j=theta,
                    problem=problem, bc_choice=bc_choice, quad_rule=quad_rule, use_nudft=use_nudft
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
                    label=f"Uniform (N={N_unif})" if r_s == R else "")
        ax2.scatter(x_ad, y_ad, s=24, color='crimson', marker='^', alpha=0.85,
                    label=f"Adapted Clustered (N={N_adapt})" if r_s == R else "")

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



def plot_3x3_disk_error_comparison(problem, N_list=[32, 64, 128], M=64,
                                   cluster_strength=0.40, theta_0=np.pi,
                                   quad_rule=2, bc_choice=1):
    """
    Renders a 3x3 plot comparing Pointwise Error Surfaces on the disk:
    Rows: N = 32, N = 64, N = 128
    Columns: Adapted NUFFT | Adapted NUDFT | Uniform FFT
    """
    fig = plt.figure(figsize=(11, 8.5))
    methods = [
        ("Adapted NUFFT", 1, False),
        ("Adapted NUDFT", 1, True),
        ("Uniform FFT", 2, False)
    ]

    plot_idx = 1
    for row_i, N in enumerate(N_list):
        for col_j, (meth_name, azu_unif, use_nudft) in enumerate(methods):
            if azu_unif == 1:
                theta = generate_adapted_clustered_azimuthal(N, cluster_strength=cluster_strength, center=theta_0)
            else:
                theta = generate_uniform_azimuthal(N)

            res = run_benchmark_case(
                N=N, M=M, azu_unif=azu_unif, theta_j=theta,
                problem=problem, bc_choice=bc_choice, quad_rule=quad_rule, use_nudft=use_nudft
            )

            ax = fig.add_subplot(len(N_list), 3, plot_idx, projection='3d')
            err = np.abs(res["u_true"] - res["u_approx"])
            X = res["x_coord"]
            Y = res["y_coord"]

            Xp = np.vstack([X, X[0, :]])
            Yp = np.vstack([Y, Y[0, :]])
            Ep = np.vstack([err, err[0, :]])

            surf = ax.plot_surface(Xp, Yp, Ep, cmap='inferno', edgecolor='none')
            ax.set_title(f"{meth_name} (N={N}, M={M})\n$L_2$ Error = {res['L2_rel']:.2e}", fontsize=8)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)
            ax.set_zlabel("Error", fontsize=8)
            fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, pad=0.1)

            plot_idx += 1

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
    th_un_high = generate_uniform_azimuthal(N_unif_high)
    res_unif = run_benchmark_case(
        N=N_unif_high, M=M, azu_unif=2, theta_j=th_un_high, problem=problem,
        bc_choice=bc_choice, quad_rule=quad_rule, use_nudft=False
    )

    cases = [
        (f"Adapted NUFFT (N={N_adapt}, M={M})", res_nufft),
        (f"Adapted NUDFT (N={N_adapt}, M={M})", res_nudft),
        (f"Uniform FFT High-Res (N={N_unif_high}, M={M})", res_unif)
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
        "Uniform FFT",
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
        "Uniform FFT": "royalblue",
    }
    markers = {
        "Adapted NUFFT": "s-",
        "Adapted NUDFT": "^-",
        "Uniform FFT": "o-",
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
        "Uniform FFT": "royalblue",
    }
    markers = {
        "Adapted NUFFT": "s--",
        "Adapted NUDFT": "^-.",
        "Uniform FFT": "o-",
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

