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
def get_azimuthal_benchmark_problem(R=1.0, theta_0=np.pi, k_mode=4, gamma=0.65):
    """
    Constructs an exact analytical Poisson problem on the disk B(0, R):

        u(r, theta) = r^2 * (R^2 - r^2) * h(theta)
        h(theta) = cos(k * theta) / (1 - gamma * cos(theta - theta_0))

    This solution features:
    - Homogeneous Dirichlet boundary condition: u(R, theta) = 0.
    - Smooth behavior at origin r -> 0: u(0, theta) = 0.
    - Localized high-frequency angular front around theta = theta_0.
    - Exact Laplacian:
        Δu = (4*R^2 - 16*r^2) * h(theta) + (R^2 - r^2) * h''(theta)
    """
    th_sym = sp.symbols('th', real=True)
    h_sym = sp.cos(k_mode * th_sym) / (1 - gamma * sp.cos(th_sym - theta_0))
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

    def g_dirichlet(xc, yc):
        return u_true(xc, yc)

    def g_neumann(xc, yc, R_val=R):
        rc = np.sqrt(xc**2 + yc**2)
        thc = np.arctan2(yc, xc)
        return -2.0 * (R_val**3) * h_func(thc)

    return {
        "u": u_true,
        "f": f_rhs,
        "g_dirichlet": g_dirichlet,
        "g_neumann": g_neumann,
        "h_func": h_func,
        "R": R,
        "theta_0": theta_0,
        "k_mode": k_mode,
        "gamma": gamma
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
            rad_unif=1,               # Radial grid is always uniform
            azu_unif=azu_unif,
            use_nudft_angular=use_nudft,
            maxiter_nufft=maxiter_nufft,
            tol_nufft=tol_nufft
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
# Table Rendering Utilities (N vs M Tables for Each Algorithm)
# ---------------------------------------------------------
def render_algorithm_tables(df_results, value_col="L2_rel", format_sci=True):
    """
    Renders clean N vs M HTML tables for each algorithm.
    Rows: N (Angular points), Columns: M (Radial points).
    """
    methods = ["Adapted NUFFT", "Adapted NUDFT", "Uniform FFT"]
    tables = {}

    for meth in methods:
        sub_df = df_results[df_results["method"] == meth]
        if sub_df.empty:
            continue
        pivot = sub_df.pivot(index="N", columns="M", values=value_col)
        if format_sci:
            formatted = pivot.map(lambda v: f"{v:.2e}")
        else:
            formatted = pivot.map(lambda v: f"{v:.4f}")

        col_title = "Relative L₂ Error" if value_col == "L2_rel" else ("Relative L_∞ Error" if value_col == "Linf_rel" else "Runtime (s)")
        print(f"\n{'='*75}\n{meth} — {col_title} (N vs M)\n{'='*75}")
        display(HTML(formatted.to_html(classes="table table-bordered table-striped text-center")))
        tables[meth] = pivot

    return tables


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

    fig = plt.figure(figsize=(18, 5))

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
    ax3.set_xlabel(r"Azimuthal Angle $\theta$ (rad)", fontsize=11)
    ax3.set_ylabel(r"$u(0.7R, \theta)$", fontsize=11)
    ax3.set_title("Node Density over Sharp Angular Wave", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_accuracy_and_runtimes_from_df(df_results, fixed_M=64, fixed_N=64):
    """
    Plots:
    1. L2 Error vs. N (fixed M) for all 3 methods.
    2. L2 Error vs. M (fixed N) for all 3 methods.
    3. Solver Runtime vs. N for all 3 methods.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"Adapted NUFFT": "crimson", "Adapted NUDFT": "forestgreen", "Uniform FFT": "royalblue"}
    markers = {"Adapted NUFFT": "s--", "Adapted NUDFT": "^-.", "Uniform FFT": "o-"}

    # Subplot 1: Error vs N (fixed M)
    sub_M = df_results[df_results["M"] == fixed_M]
    for meth, grp in sub_M.groupby("method"):
        grp_s = grp.sort_values("N")
        ax1.loglog(grp_s["N"], grp_s["L2_rel"], markers.get(meth, "o-"),
                   color=colors.get(meth, "black"), lw=2.2, ms=6, label=meth)

    ax1.set_xlabel("Angular Points $N$", fontsize=11)
    ax1.set_ylabel(r"Relative $L_2$ Error", fontsize=11)
    ax1.set_title(f"Accuracy vs. $N$ (Fixed $M={fixed_M}$)", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Subplot 2: Error vs M (fixed N)
    sub_N = df_results[df_results["N"] == fixed_N]
    for meth, grp in sub_N.groupby("method"):
        grp_s = grp.sort_values("M")
        ax2.loglog(grp_s["M"], grp_s["L2_rel"], markers.get(meth, "o-"),
                   color=colors.get(meth, "black"), lw=2.2, ms=6, label=meth)

    ax2.set_xlabel("Radial Points $M$", fontsize=11)
    ax2.set_ylabel(r"Relative $L_2$ Error", fontsize=11)
    ax2.set_title(f"Accuracy vs. $M$ (Fixed $N={fixed_N}$)", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    # Subplot 3: Runtime vs N
    for meth, grp in sub_M.groupby("method"):
        grp_s = grp.sort_values("N")
        ax3.loglog(grp_s["N"], grp_s["runtime"], markers.get(meth, "o-"),
                   color=colors.get(meth, "black"), lw=2.2, ms=6, label=meth)

    ax3.set_xlabel("Angular Points $N$", fontsize=11)
    ax3.set_ylabel("Runtime (seconds)", fontsize=11)
    ax3.set_title(f"Execution Time vs. $N$ (Fixed $M={fixed_M}$)", fontsize=12)
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
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
    fig = plt.figure(figsize=(18, 14))
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
            ax.set_title(f"{meth_name} (N={N}, M={M})\n$L_2$ Error = {res['L2_rel']:.2e}", fontsize=10)
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
    fig = plt.figure(figsize=(18, 5))

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
        ax.set_title(f"{title}\n$L_2$ Error = {res['L2_rel']:.2e} | t = {res['runtime']:.4f}s", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Pointwise Error")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.show()
