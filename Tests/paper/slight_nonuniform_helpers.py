import os
import sys
from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, HTML
from tqdm.auto import tqdm
from scipy.interpolate import CubicSpline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_jittered_azimuthal,
    generate_cartesian_grid_on_disk,
    compute_zero_mode,
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver


# ==============================================================================
# Multi-Run Timing Configuration & Global Backend
# ==============================================================================
_GLOBAL_TIME_TRIALS = False
_GLOBAL_NUM_RUNS = 5
_GLOBAL_USE_GPU = False

def set_timing_config(time_trials=False, num_runs=5, use_gpu=False):
    """Globally configure multi-trial benchmark timing and default backend."""
    global _GLOBAL_TIME_TRIALS, _GLOBAL_NUM_RUNS, _GLOBAL_USE_GPU
    _GLOBAL_TIME_TRIALS = bool(time_trials)
    _GLOBAL_NUM_RUNS = int(num_runs) if time_trials else 1
    _GLOBAL_USE_GPU = bool(use_gpu)


def get_single_multipole_problem(R=1.0, mode=12, theta_0=np.pi):
    """Smooth Dirichlet disk problem u=(R^2-r^2)(r/R)^m cos(m(theta-theta_0))."""
    if not isinstance(mode, (int, np.integer)) or mode < 1:
        raise ValueError("mode must be a positive integer")

    def q(x, y):
        return (np.hypot(x, y) / R) ** mode

    def c(x, y):
        return np.cos(mode * (np.arctan2(y, x) - theta_0))

    def u(x, y):
        return (R**2 - x**2 - y**2) * q(x, y) * c(x, y)

    def f(x, y):
        return -4.0 * (mode + 1) * q(x, y) * c(x, y)

    def g_dir(x, y):
        return np.zeros_like(x)

    def g_neu(x, y, R_val=R):
        return -2.0 * R_val * np.cos(mode * (np.arctan2(y, x) - theta_0))

    return {"u": u, "f": f, "g_dirichlet": g_dir, "g_neumann": g_neu,
            "R": R, "mode": mode, "theta_0": theta_0}

def get_cartesian_mixed_problem(R=1.0, k=8):
    """
    Smooth nonseparable manufactured Dirichlet problem on a disk.

    Exact solution:
        u(x, y) = (R^2 - x^2 - y^2) exp(x) sin(k y)

    This implementation assumes R = 1 for the analytical forcing
    written below.
    """
    if R != 1.0:
        raise ValueError(
            "get_cartesian_mixed_problem currently assumes R = 1.0"
        )

    def u(x, y):
        return (
            (1.0 - x**2 - y**2)
            * np.exp(x)
            * np.sin(k * y)
        )

    def f(x, y):
        exp_x = np.exp(x)
        sin_ky = np.sin(k * y)
        cos_ky = np.cos(k * y)

        return -(
            (k**2 - 1.0)
            * (1.0 - x**2 - y**2)
            * exp_x
            * sin_ky
            + 4.0 * exp_x * sin_ky
            + 4.0 * x * exp_x * sin_ky
            + 4.0 * k * y * exp_x * cos_ky
        )

    def g_dir(x, y):
        return np.zeros_like(x)

    def g_neu(x, y, R_val=R):
        raise NotImplementedError(
            "This benchmark is intended for Dirichlet conditions."
        )

    return {
        "u": u,
        "f": f,
        "g_dirichlet": g_dir,
        "g_neumann": g_neu,
        "R": R,
        "mode": None,
        "theta_0": None,
    }


def generate_jittered_azimuthal_fixed(N, jitter_fraction=0.35, seed=42):
    """Reproducible jittered azimuthal mesh for benchmark comparisons."""
    np.random.seed(seed)
    return generate_jittered_azimuthal(N, jitter_fraction=jitter_fraction)


def periodic_linear_interpolate(theta_src, values_src, theta_tgt):
    """Periodic periodic cubic spline; angular samples occupy axis 0."""
    theta_src = np.asarray(theta_src, dtype=float)
    values_src = np.asarray(values_src)
    if theta_src.ndim != 1 or len(theta_src) != values_src.shape[0]:
        raise ValueError("theta_src must be 1-D and match values_src axis 0")
    theta_ext = np.r_[theta_src[-1] - 2*np.pi, theta_src, theta_src[0] + 2*np.pi]
    values_ext = np.concatenate((values_src[-1:], values_src, values_src[:1]), axis=0)
    if values_src.ndim == 1:
        return np.interp(theta_tgt, theta_ext, values_ext)
    return np.column_stack([np.interp(theta_tgt, theta_ext, values_ext[:, k])
                            for k in range(values_src.shape[1])])

def periodic_cubic_spline_interpolate(
    theta_src,
    values_src,
    theta_tgt,
):
    """
    Periodic cubic-spline interpolation from nonuniform angular samples
    onto target angular positions.

    theta_src:
        Sorted source angular angles, shape (P,).

    values_src:
        Values sampled at theta_src, shape (P,) or (P, M).

    theta_tgt:
        Target angles, generally an equispaced Uniform FFT grid.
    """
    theta_src = np.asarray(
        theta_src,
        dtype=float,
    )

    values_src = np.asarray(
        values_src,
    )

    if theta_src.ndim != 1:
        raise ValueError("theta_src must be one-dimensional")

    if values_src.shape[0] != len(theta_src):
        raise ValueError(
            "values_src must have len(theta_src) entries on axis 0"
        )

    # Append a periodic copy of the first point.
    theta_extended = np.r_[
        theta_src,
        theta_src[0] + 2.0 * np.pi,
    ]

    values_extended = np.concatenate(
        [
            values_src,
            values_src[:1],
        ],
        axis=0,
    )

    # Place all evaluation angles in the spline's periodic interval.
    theta_target_wrapped = np.mod(
        theta_tgt - theta_src[0],
        2.0 * np.pi,
    ) + theta_src[0]

    spline = CubicSpline(
        theta_extended,
        values_extended,
        axis=0,
        bc_type="periodic",
    )

    return spline(theta_target_wrapped)

def run_benchmark_case(
    N,
    M,
    azu_unif,
    theta_j,
    problem,
    bc_choice=1,
    quad_rule=2,
    use_nudft=False,
    maxiter_nufft=100,
    tol_nufft=1e-10,
    reg_param=1e-12,
    eps_finufft=1e-12,
    precond_shift=1e-3,
    kde_oversample=4,
    kde_bandwidth=1.0,
    num_processors=None,
    use_gpu=None,
    time_trials=None,
    num_runs=None,
    **kwargs,
):
    """
    Solve the disk Poisson problem from jittered angular measurements.

    Timed work:
        - Uniform-grid interpolation, when azu_unif == 2.
        - Poisson solver execution.

    Untimed work:
        - Analytical forcing evaluation.
        - Exact solution evaluation.
        - Error metric calculation.
        - Plot-data preparation.
    """
    actual_use_gpu = _GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)
    R = problem["R"]

    # --------------------------------------------------------------
    # Build the radial grid and acquire source measurements.
    # These steps are excluded from timing for every method.
    # --------------------------------------------------------------
    r_m = generate_uniform_radial(M, R)

    theta_raw = np.asarray(
        theta_j,
        dtype=float,
    )

    x_raw, y_raw = generate_cartesian_grid_on_disk(
        theta_raw,
        r_m,
    )

    f_raw = problem["f"](
        x_raw,
        y_raw,
    )

    if bc_choice == 1:
        g_raw = problem["g_dirichlet"](
            x_raw[:, -1],
            y_raw[:, -1],
        )
    else:
        g_raw = problem["g_neumann"](
            x_raw[:, -1],
            y_raw[:, -1],
            R,
        )

    # --------------------------------------------------------------
    # Homogeneous Dirichlet boundary condition:
    #
    # u(R, theta) = 0, so the boundary zero Fourier mode is exactly 0.
    #
    # Uniform FFT retains its existing empty-array convention.
    # --------------------------------------------------------------
    if bc_choice == 1:
        if azu_unif == 1:
            u_fourier_0 = 0.0
        else:
            u_fourier_0 = np.array([])

    else:
        # Preserve the optional Neumann path.
        # This is outside the main Dirichlet experiment.
        theta_for_bc = generate_uniform_azimuthal(N) if azu_unif == 2 else theta_raw
        x_bc, y_bc = generate_cartesian_grid_on_disk(
            theta_for_bc,
            r_m,
        )

        u_bc = problem["u"](
            x_bc,
            y_bc,
        )

        u_fourier_0 = compute_zero_mode(
            u_bc,
            theta_for_bc,
            azu_unif,
        )[-1]

    # --------------------------------------------------------------
    # Timed Poisson solve.
    # --------------------------------------------------------------
    if time_trials is not None:
        effective_time_trials = bool(time_trials)
    else:
        effective_time_trials = _GLOBAL_TIME_TRIALS

    if num_runs is not None:
        n_runs = int(num_runs)
    else:
        n_runs = _GLOBAL_NUM_RUNS if effective_time_trials else 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        runtimes = []
        for _ in range(n_runs):
            if actual_use_gpu:
                try:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()
                except Exception:
                    pass

            t0 = time.perf_counter()

            if azu_unif == 2:
                # Uniform FFT pipeline:
                # jittered measurements -> periodic cubic-spline interpolation ->
                # uniform angular target grid.
                theta_solver = generate_uniform_azimuthal(N)
                f_values = periodic_cubic_spline_interpolate(
                    theta_raw,
                    f_raw,
                    theta_solver,
                )
                g_values = periodic_cubic_spline_interpolate(
                    theta_raw,
                    g_raw,
                    theta_solver,
                )
            else:
                # Direct NUFFT / NUDFT pipeline:
                # consume jittered angular measurements directly.
                theta_solver = theta_raw
                f_values = f_raw
                g_values = g_raw

            u_approx = poisson_solver(
                f_values=f_values,
                g_values=g_values,
                u_fourier_0=u_fourier_0,
                N=N,
                M=M,
                r_m=r_m,
                theta_j=theta_solver,
                R=R,
                quad_rule=quad_rule,
                BC_choice=bc_choice,
                rad_unif=1,
                grid_type=(1 if azu_unif == 2 else 2),
                use_nudft_angular=use_nudft,
                maxiter_nufft=maxiter_nufft,
                tol_nufft=tol_nufft,
                reg_param=reg_param,
                eps_finufft=eps_finufft,
                precond_shift=precond_shift,
                kde_oversample=kde_oversample,
                kde_bandwidth=kde_bandwidth,
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

    # --------------------------------------------------------------
    # Evaluate exact solution and errors outside the timed region.
    # --------------------------------------------------------------
    x_sol, y_sol = generate_cartesian_grid_on_disk(
        theta_solver,
        r_m,
    )

    u_true = problem["u"](
        x_sol,
        y_sol,
    )

    _, linf_rel, _, l2_rel = compute_error_metrics(
        u_approx,
        u_true,
        r_m,
        theta_solver,
    )

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
        "u_true": u_true,
        "x_coord": x_sol,
        "y_coord": y_sol,
        "iRadius": r_m,
        "iAngle": theta_solver,
    }


def run_all_algorithms_NxM_study(
    problem,
    N_values,
    M_values,
    jitter_fraction=0.35,
    grid_seed=42,
    bc_choice=1,
    quad_rule=2,
    maxiter_nufft=100,
    tol_nufft=1e-10,
    reg_param=1e-12,
    eps_finufft=1e-12,
    precond_shift=1e-3,
    kde_oversample=4,
    kde_bandwidth=1.0,
    num_processors=None,
    use_gpu=None,
    time_trials=None,
    num_runs=None,
    **kwargs,
):
    algorithms = [
        ("Adapted NUFFT", 1, False),
        ("Adapted NUDFT", 1, True),
        ("Uniform FFT + periodic cubic spline", 2, False),
    ]
    actual_use_gpu = _GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)
    # Comprehensive Warmup Solve across ALL algorithms (Warms up cuSOLVER, cuFFT, cuFINUFFT plans, and CPU caches)
    for _, w_azu, w_nudft in algorithms:
        try:
            th_warmup = generate_jittered_azimuthal_fixed(N_values[0], jitter_fraction, grid_seed)
            run_benchmark_case(
                N_values[0],
                M_values[0],
                w_azu,
                th_warmup,
                problem,
                bc_choice,
                quad_rule,
                w_nudft,
                maxiter_nufft=maxiter_nufft,
                tol_nufft=tol_nufft,
                reg_param=reg_param,
                eps_finufft=eps_finufft,
                precond_shift=precond_shift,
                kde_oversample=kde_oversample,
                kde_bandwidth=kde_bandwidth,
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
    pbar = tqdm(total=len(N_values)*len(M_values)*len(algorithms),
                desc=f"Computing N x M jittered-grid study [{backend_label}]")
    for N in N_values:
        theta = generate_jittered_azimuthal_fixed(N, jitter_fraction, grid_seed)
        for M in M_values:
            for name, azu_unif, use_nudft in algorithms:
                row = run_benchmark_case(
                    N,
                    M,
                    azu_unif,
                    theta,
                    problem,
                    bc_choice,
                    quad_rule,
                    use_nudft,
                    maxiter_nufft=maxiter_nufft,
                    tol_nufft=tol_nufft,
                    reg_param=reg_param,
                    eps_finufft=eps_finufft,
                    precond_shift=precond_shift,
                    kde_oversample=kde_oversample,
                    kde_bandwidth=kde_bandwidth,
                    num_processors=num_processors,
                    use_gpu=actual_use_gpu,
                    time_trials=time_trials,
                    num_runs=num_runs,
                    **kwargs,
                )
                row["method"] = name
                rows.append(row)
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def plot_solution_and_grids(problem, N_adapt=32, N_unif=32, M=8,
                            jitter_fraction=0.35, grid_seed=42):
    """
    Compact journal-style 1 x 3 visualization:
      (a) exact solution on the disk,
      (b) jittered polar measurement grid,
      (c) uniform polar FFT target grid.

    Parameters
    ----------
    N_adapt : int
        Number of jittered azimuthal spokes.
    N_unif : int
        Number of uniform azimuthal spokes.
    M : int
        Number of radial rings, including the outer boundary.
    """
    R = problem["R"]

    # Angular positions for the two polar grids.
    theta_adapt = generate_jittered_azimuthal_fixed(
        N_adapt, jitter_fraction, grid_seed
    )
    theta_unif = generate_uniform_azimuthal(N_unif)

    # M radial rings, including the outer boundary r = R.
    radii = np.linspace(R / M, R, M)

    # Fine disk grid for the exact-solution surface.
    fine_theta = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    fine_r = np.linspace(0, R, 160)
    Xf, Yf = generate_cartesian_grid_on_disk(fine_theta, fine_r)
    Uf = problem["u"](Xf, Yf)

    # Use 8-point font consistently.
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Compact one-row, three-column figure.
    fig = plt.figure(figsize=(12, 3.8), dpi=150)

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.22, 1, 1],
        left=0.045,
        right=0.985,
        bottom=0.12,
        top=0.84,
        wspace=0.20,
    )

    # ===============================================================
    # (a) Exact solution
    # ===============================================================
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")

    surf = ax0.plot_surface(
        Xf,
        Yf,
        Uf,
        cmap="plasma",
        edgecolor="none",
        antialiased=True,
        rcount=120,
        ccount=180,
    )

    ax0.set_title(
        r"(a) Exact manufactured solution",
        fontsize=8,
        pad=8,
        fontweight="semibold",
    )

    ax0.set_xlabel(r"$x$", fontsize=8, labelpad=2)
    ax0.set_ylabel(r"$y$", fontsize=8, labelpad=2)
    ax0.set_zlabel("")

    ax0.set_xlim(-R, R)
    ax0.set_ylim(-R, R)
    ax0.view_init(elev=27, azim=-52)

    # Make the 3D panel compact without crowding the other subplots.
    ax0.set_box_aspect((1, 1, 0.58), zoom=1.00)

    ax0.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax0.set_yticks([-1, -0.5, 0, 0.5, 1])

    # The colorbar provides the solution scale; omit z-axis tick labels.
    ax0.set_zticks([])

    ax0.tick_params(axis="x", labelsize=8, pad=1)
    ax0.tick_params(axis="y", labelsize=8, pad=1)

    cbar = fig.colorbar(
        surf,
        ax=ax0,
        shrink=0.62,
        aspect=22,
        pad=0.14,
    )

    cbar.set_label(
        r"$u(x,y)$",
        rotation=270,
        labelpad=8,
        fontsize=8,
    )
    cbar.ax.tick_params(labelsize=8, pad=2)

    # ===============================================================
    # Helper for polar grid panels
    # ===============================================================
    def draw_polar_grid(ax, theta, color, panel_title):
        circle_theta = np.linspace(0, 2 * np.pi, 600)

        disk = plt.Circle(
            (0, 0),
            R,
            facecolor="0.985",
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(disk)

        # Concentric radial rings.
        for j, r in enumerate(radii):
            is_boundary = j == len(radii) - 1

            ax.plot(
                r * np.cos(circle_theta),
                r * np.sin(circle_theta),
                color="0.50" if is_boundary else "0.72",
                lw=1.15 if is_boundary else 0.85,
                zorder=1,
            )

        # Azimuthal spokes.
        for angle in theta:
            ax.plot(
                [0, R * np.cos(angle)],
                [0, R * np.sin(angle)],
                color=color,
                lw=0.85,
                alpha=0.68,
                zorder=2,
            )

        # Nodes at all ring/spoke intersections.
        R_mesh, Theta_mesh = np.meshgrid(radii, theta)
        x_nodes = R_mesh * np.cos(Theta_mesh)
        y_nodes = R_mesh * np.sin(Theta_mesh)

        ax.scatter(
            x_nodes.ravel(),
            y_nodes.ravel(),
            s=24,
            color=color,
            edgecolors="white",
            linewidths=0.55,
            zorder=4,
        )

        # Center node.
        ax.scatter(
            0,
            0,
            s=32,
            color=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=5,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.10 * R, 1.10 * R)
        ax.set_ylim(-1.10 * R, 1.10 * R)

        ax.set_title(
            panel_title,
            fontsize=8,
            pad=6,
            fontweight="semibold",
            linespacing=1.1,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    # ===============================================================
    # (b) Jittered measurement grid
    # ===============================================================
    ax1 = fig.add_subplot(gs[0, 1])

    draw_polar_grid(
        ax=ax1,
        theta=theta_adapt,
        color="#C6284A",
        panel_title=(
            r"(b) Jittered measurement grid"
            "\n"
            rf"$N_\theta={N_adapt}, \quad N_r={M}$"
        ),
    )

    # ===============================================================
    # (c) Uniform FFT target grid
    # ===============================================================
    ax2 = fig.add_subplot(gs[0, 2])

    draw_polar_grid(
        ax=ax2,
        theta=theta_unif,
        color="#3268B8",
        panel_title=(
            r"(c) Uniform FFT target grid"
            "\n"
            rf"$N_\theta={N_unif}, \quad N_r={M}$"
        ),
    )

    plt.show()

def _plot_error_surface(ax, res, title):
    err = np.abs(res["u_true"] - res["u_approx"])
    X, Y = res["x_coord"], res["y_coord"]
    surf = ax.plot_surface(np.vstack((X, X[0])), np.vstack((Y, Y[0])),
                           np.vstack((err, err[0])), cmap="inferno", edgecolor="none")
    ax.set_title(f"{title}\n$L_2$={res['L2_rel']:.2e}", fontsize=8)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8); ax.set_zlabel("error", fontsize=8)
    return surf


def plot_adapted_vs_highres_uniform(
    problem,
    N_adapt=32,
    N_unif_high=128,
    M=64,
    jitter_fraction=0.35,
    grid_seed=42,
    quad_rule=2,
    bc_choice=1,
    maxiter_nufft=100,
    tol_nufft=1e-10,
    reg_param=1e-12,
    eps_finufft=1e-12,
    precond_shift=1e-3,
    kde_oversample=4,
    kde_bandwidth=1.0,
    num_processors=None,
    use_gpu=None,
    time_trials=None,
    num_runs=None,
    **kwargs,
):
    """
    Render a 1 x 3 disk-error comparison.

    Plots:
      (a) Adapted NUFFT using N_adapt jittered measurements directly.
      (b) Uniform FFT + interpolation using the same N_adapt
          jittered measurements.
      (c) Uniform FFT + interpolation using N_unif_high
          jittered measurements.

    Table:
      Includes Adapted NUFFT, Adapted NUDFT, equal-budget Uniform FFT,
      and higher-budget Uniform FFT results.
    """
    # --------------------------------------------------------------
    # Low-resolution jittered measurement grid:
    # shared by NUFFT, NUDFT, and equal-budget Uniform FFT.
    # --------------------------------------------------------------
    theta_low = generate_jittered_azimuthal_fixed(
        N_adapt,
        jitter_fraction=jitter_fraction,
        seed=grid_seed,
    )

    # --------------------------------------------------------------
    # High-resolution jittered measurement grid:
    # used only by the higher-budget Uniform FFT comparison.
    # --------------------------------------------------------------
    theta_high = generate_jittered_azimuthal_fixed(
        N_unif_high,
        jitter_fraction=jitter_fraction,
        seed=grid_seed,
    )

    # --------------------------------------------------------------
    # Case 1: Adapted NUFFT directly uses jittered samples.
    # --------------------------------------------------------------
    res_nufft = run_benchmark_case(
        N=N_adapt,
        M=M,
        azu_unif=1,
        theta_j=theta_low,
        problem=problem,
        bc_choice=bc_choice,
        quad_rule=quad_rule,
        use_nudft=False,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps_finufft=eps_finufft,
        precond_shift=precond_shift,
        kde_oversample=kde_oversample,
        kde_bandwidth=kde_bandwidth,
        num_processors=num_processors,
        use_gpu=use_gpu,
        time_trials=time_trials,
        num_runs=num_runs,
        **kwargs,
    )

    # --------------------------------------------------------------
    # Case 2: Adapted NUDFT directly uses the same jittered samples.
    # This case appears in the table, but not in the 1 x 3 plot.
    # --------------------------------------------------------------
    res_nudft = run_benchmark_case(
        N=N_adapt,
        M=M,
        azu_unif=1,
        theta_j=theta_low,
        problem=problem,
        bc_choice=bc_choice,
        quad_rule=quad_rule,
        use_nudft=True,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps_finufft=eps_finufft,
        precond_shift=precond_shift,
        kde_oversample=kde_oversample,
        kde_bandwidth=kde_bandwidth,
        num_processors=num_processors,
        use_gpu=use_gpu,
        time_trials=time_trials,
        num_runs=num_runs,
        **kwargs,
    )

    # --------------------------------------------------------------
    # Case 3: Uniform FFT using the same low-N measurement budget.
    # --------------------------------------------------------------
    res_uniform_equal = run_benchmark_case(
        N=N_adapt,
        M=M,
        azu_unif=2,
        theta_j=theta_low,
        problem=problem,
        bc_choice=bc_choice,
        quad_rule=quad_rule,
        use_nudft=False,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps_finufft=eps_finufft,
        precond_shift=precond_shift,
        kde_oversample=kde_oversample,
        kde_bandwidth=kde_bandwidth,
        num_processors=num_processors,
        use_gpu=use_gpu,
        time_trials=time_trials,
        num_runs=num_runs,
        **kwargs,
    )

    # --------------------------------------------------------------
    # Case 4: Uniform FFT using the higher measurement budget.
    # --------------------------------------------------------------
    res_uniform_high = run_benchmark_case(
        N=N_unif_high,
        M=M,
        azu_unif=2,
        theta_j=theta_high,
        problem=problem,
        bc_choice=bc_choice,
        quad_rule=quad_rule,
        use_nudft=False,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps_finufft=eps_finufft,
        precond_shift=precond_shift,
        kde_oversample=kde_oversample,
        kde_bandwidth=kde_bandwidth,
        num_processors=num_processors,
        use_gpu=use_gpu,
        time_trials=time_trials,
        num_runs=num_runs,
        **kwargs,
    )

    # --------------------------------------------------------------
    # Three cases shown in the 1 x 3 visualization.
    # --------------------------------------------------------------
    plot_cases = [
        (
            "Adapted NUFFT\n"
            "direct jittered data",
            res_nufft,
        ),
        (
            "Uniform FFT + interpolation\n"
            "same measurement budget",
            res_uniform_equal,
        ),
        (
            "Uniform FFT + interpolation\n"
            "higher measurement budget",
            res_uniform_high,
        ),
    ]

    # --------------------------------------------------------------
    # Four cases listed in the summary table.
    # --------------------------------------------------------------
    table_cases = [
        (
            "Adapted NUFFT — direct jittered data",
            res_nufft,
        ),
        (
            "Adapted NUDFT — direct jittered data",
            res_nudft,
        ),
        (
            "Uniform FFT + interpolation — same measurement budget",
            res_uniform_equal,
        ),
        (
            "Uniform FFT + interpolation — higher measurement budget",
            res_uniform_high,
        ),
    ]

    # --------------------------------------------------------------
    # Numerical summary table, including NUDFT.
    # --------------------------------------------------------------
    summary_rows = []

    for label, res in table_cases:
        summary_rows.append({
            "Case": label,
            "N": res["N"],
            "M": res["M"],
            "Relative L2": res["L2_rel"],
            "Relative Linf": res["Linf_rel"],
            "Runtime (s)": res["runtime"],
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 110)
    print("1 x 3 Disk Error Comparison with NUDFT Table Result")
    print("=" * 110)

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "Relative L2": lambda value: f"{value:.3e}",
                "Relative Linf": lambda value: f"{value:.3e}",
                "Runtime (s)": lambda value: f"{value:.4f}",
            },
        )
    )

    # --------------------------------------------------------------
    # Compute pointwise errors only for the three plotted cases.
    # --------------------------------------------------------------
    all_errors = [
        np.abs(res["u_true"] - res["u_approx"])
        for _, res in plot_cases
    ]

    error_max = max(np.max(error) for error in all_errors)

    if error_max <= 0:
        error_max = 1.0

    # --------------------------------------------------------------
    # Compact journal-style formatting.
    # --------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig = plt.figure(figsize=(12, 3.8), dpi=150)

    gs = fig.add_gridspec(
        1,
        3,
        left=0.04,
        right=0.90,
        bottom=0.08,
        top=0.84,
        wspace=0.22,
    )

    surfaces = []

    # --------------------------------------------------------------
    # Render the three requested error surfaces.
    # --------------------------------------------------------------
    for plot_index, ((label, res), error) in enumerate(
        zip(plot_cases, all_errors)
    ):
        ax = fig.add_subplot(
            gs[0, plot_index],
            projection="3d",
        )

        X = res["x_coord"]
        Y = res["y_coord"]

        # Repeat the first angular row to close the periodic disk rim.
        X_plot = np.vstack([X, X[0, :]])
        Y_plot = np.vstack([Y, Y[0, :]])
        error_plot = np.vstack([error, error[0, :]])

        surface = ax.plot_surface(
            X_plot,
            Y_plot,
            error_plot,
            cmap="inferno",
            edgecolor="none",
            antialiased=True,
            vmin=0.0,
            vmax=error_max,
            rcount=min(160, X_plot.shape[0]),
            ccount=min(160, X_plot.shape[1]),
        )

        surfaces.append(surface)

        ax.set_xlim(-problem["R"], problem["R"])
        ax.set_ylim(-problem["R"], problem["R"])
        ax.set_zlim(0.0, error_max)

        ax.view_init(elev=28, azim=-52)
        ax.set_box_aspect((1, 1, 0.55), zoom=1.0)

        ax.set_title(
            f"({chr(97 + plot_index)}) {label}\n"
            f"$N={res['N']}, \\; M={res['M']}$\n"
            f"$L_2={res['L2_rel']:.2e}$, "
            f"$L_\\infty={res['Linf_rel']:.2e}$",
            fontsize=8,
            pad=8,
            fontweight="semibold",
        )

        ax.set_xlabel(r"$x$", fontsize=8, labelpad=1)
        ax.set_ylabel(r"$y$", fontsize=8, labelpad=1)

        # The shared colorbar communicates the error scale.
        ax.set_zticks([])

        ax.tick_params(axis="x", labelsize=7, pad=0)
        ax.tick_params(axis="y", labelsize=7, pad=0)

    # --------------------------------------------------------------
    # Shared colorbar for the three plotted surfaces.
    # --------------------------------------------------------------
    cbar = fig.colorbar(
        surfaces[0],
        ax=fig.axes[:3],
        shrink=0.70,
        aspect=20,
        pad=0.04,
    )

    cbar.set_label(
        "Pointwise error",
        rotation=270,
        labelpad=10,
        fontsize=8,
    )
    cbar.ax.tick_params(labelsize=7, pad=2)

    plt.show()

    return summary_df
      
def render_combined_runtime_table(df_results):
    pivot = df_results.pivot_table(index="N", columns=["method", "M"], values="runtime").sort_index(axis=1)
    display(HTML(pivot.map(lambda v: f"{v:.4f}" if np.isfinite(v) else "—").to_html(classes="table table-bordered text-center")))
    return pivot


def render_combined_error_table(df_results, value_col="L2_rel"):
    order = ["Adapted NUFFT", "Adapted NUDFT", "Uniform FFT + periodic cubic spline"]
    pivot = df_results.pivot_table(index="N", columns=["method", "M"], values=value_col, aggfunc="first")
    columns = [(name, M) for name in order for M in sorted(df_results["M"].unique()) if (name, M) in pivot.columns]
    pivot = pivot.reindex(columns=columns).sort_index()
    display(HTML(pivot.map(lambda v: f"{v:.2e}" if np.isfinite(v) else "—").to_html(classes="table table-bordered text-center")))
    return pivot


def plot_accuracy_from_df(df_results, fixed_M=64, fixed_N=64, title_prefix=None):
    colors = {
        "Adapted NUFFT": "crimson",
        "Adapted NUDFT": "forestgreen",
        "Uniform FFT + periodic cubic spline": "royalblue",
        "Adapted / NUFFT": "crimson",
        "Adapted / NUDFT": "forestgreen",
        "Uniform / FFT + cubic spline": "royalblue",
    }
    markers = {
        "Adapted NUFFT": "s--",
        "Adapted NUDFT": "^-.",
        "Uniform FFT + periodic cubic spline": "o-",
        "Adapted / NUFFT": "s--",
        "Adapted / NUDFT": "^-.",
        "Uniform / FFT + cubic spline": "o-",
    }
    labels = {
        "Adapted NUFFT": "Adapted / NUFFT",
        "Adapted NUDFT": "Adapted / NUDFT",
        "Uniform FFT + periodic cubic spline": "Uniform / FFT + cubic spline",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    title = title_prefix if title_prefix is not None else "Algorithm Accuracy, Slight Nonuniform Mesh"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    col = "label" if "label" in df_results.columns else "method"
    for m, g in df_results[df_results.M == fixed_M].groupby(col):
        g = g.sort_values("N")
        lbl = labels.get(m, m)
        axes[0].loglog(g.N, g.L2_rel, markers.get(m, markers.get(lbl, "o-")),
                       color=colors.get(m, colors.get(lbl, "black")), label=lbl)
    for m, g in df_results[df_results.N == fixed_N].groupby(col):
        g = g.sort_values("M")
        lbl = labels.get(m, m)
        axes[1].loglog(g.M, g.L2_rel, markers.get(m, markers.get(lbl, "o-")),
                       color=colors.get(m, colors.get(lbl, "black")), label=lbl)
    axes[0].set_title(f"Error vs N (M = {fixed_M})")
    axes[0].set_xlabel("Angular Grid Points (N)")
    axes[0].set_ylabel(r"Relative $L_2$ Error")
    axes[1].set_title(f"Error vs M (N = {fixed_N})")
    axes[1].set_xlabel("Radial Grid Points (M)")
    for ax in axes:
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()


def plot_accuracy_faceted_by_method(df_results, N_values=None, M_values=None, title_prefix=None):
    methods = [m for m in ["Adapted NUFFT", "Adapted NUDFT", "Uniform FFT + periodic cubic spline"] if m in set(df_results.method)]
    N_values = sorted(df_results.N.unique()) if N_values is None else N_values
    M_values = sorted(df_results.M.unique()) if M_values is None else M_values
    fig, axes = plt.subplots(2, len(methods), figsize=(4.5 * len(methods), 5.5), sharey="row", squeeze=False)
    cmap = plt.get_cmap("tab10")

    title = title_prefix if title_prefix is not None else "Algorithm Accuracy by Method, Slight Nonuniform Mesh"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    labels = {
        "Adapted NUFFT": "Adapted / NUFFT",
        "Adapted NUDFT": "Adapted / NUDFT",
        "Uniform FFT + periodic cubic spline": "Uniform / FFT + cubic spline",
    }

    for col_idx, method in enumerate(methods):
        sub = df_results[df_results.method == method]
        m_label = labels.get(method, method)
        for i, N in enumerate(N_values):
            g = sub[sub.N == N].sort_values("M")
            axes[0, col_idx].loglog(g.M, g.L2_rel, "o-", ms=3, color=cmap(i), label=f"N={N}")
        for i, M in enumerate(M_values):
            g = sub[sub.M == M].sort_values("N")
            axes[1, col_idx].loglog(g.N, g.L2_rel, "s-", ms=3, color=cmap(i), label=f"M={M}")
        axes[0, col_idx].set_title(f"{m_label} — Error vs M")
        axes[0, col_idx].set_xlabel("Radial Grid Points (M)")
        axes[1, col_idx].set_title(f"{m_label} — Error vs N")
        axes[1, col_idx].set_xlabel("Angular Grid Points (N)")
        for ax in (axes[0, col_idx], axes[1, col_idx]):
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(fontsize=7)
    axes[0, 0].set_ylabel(r"Relative $L_2$ Error")
    axes[1, 0].set_ylabel(r"Relative $L_2$ Error")
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    plt.show()


def plot_extreme_runtime_2x2(
    df_results,
    N_min=None,
    N_max=None,
    M_min=None,
    M_max=None,
    title_prefix=None,
    use_gpu=None,
):
    """
    2x2 runtime plot using only the lowest and highest N and M:
      Top-left:  runtime vs M for N = N_min
      Top-right: runtime vs M for N = N_max
      Bottom-left:  runtime vs N for M = M_min
      Bottom-right: runtime vs N for M = M_max
    All solver labels are plotted with a single shared legend placed below the 2x2 plots.
    """
    df_results = df_results.copy()
    col = "label" if "label" in df_results.columns else "method"
    if col in df_results.columns:
        df_results[col] = df_results[col].str.replace("(Unsquared)", "(PCGLS)", regex=False).str.replace("Unsquared", "PCGLS", regex=False)

    N_vals = sorted(df_results["N"].unique())
    M_vals = sorted(df_results["M"].unique())
    if len(N_vals) < 2 or len(M_vals) < 2:
        print("Need at least two distinct N and M values for extremes plot.")
        return

    N_min = N_vals[0] if N_min is None else N_min
    N_max = N_vals[-1] if N_max is None else N_max
    M_min = M_vals[0] if M_min is None else M_min
    M_max = M_vals[-1] if M_max is None else M_max

    actual_use_gpu = _GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)
    backend_str = "GPU" if actual_use_gpu else "CPU"

    if title_prefix is None or title_prefix in ("Runtime Scaling on a Jittered Angular Grid", "Table 1 (extreme N,M)"):
        title = f"Algorithm Runtimes ({backend_str}), Slight Nonuniform Mesh"
    else:
        title = title_prefix

    colors = {
        "Adapted NUFFT": "#C6284A",
        "Adapted NUDFT": "#2E7D32",
        "Uniform FFT + periodic cubic spline": "#3268B8",
        "Adapted / NUFFT": "#C6284A",
        "Adapted / NUDFT": "#2E7D32",
        "Uniform / FFT + cubic spline": "#3268B8",
    }

    labels = {
        "Adapted NUFFT": "Adapted / NUFFT",
        "Adapted NUDFT": "Adapted / NUDFT",
        "Uniform FFT + periodic cubic spline": "Uniform / FFT + cubic spline",
        "Adapted / NUFFT": "Adapted / NUFFT",
        "Adapted / NUDFT": "Adapted / NUDFT",
        "Uniform / FFT + cubic spline": "Uniform / FFT + cubic spline",
    }

    markers = {
        "Adapted NUFFT": "o",
        "Adapted NUDFT": "^",
        "Uniform FFT + periodic cubic spline": "s",
        "Adapted / NUFFT": "o",
        "Adapted / NUDFT": "^",
        "Uniform / FFT + cubic spline": "s",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    # Top-left: runtime vs M for N = N_min
    ax = axes[0, 0]
    df_sub = df_results[df_results["N"] == N_min]
    for m, group in df_sub.groupby(col):
        group = group.sort_values("M")
        lbl = labels.get(m, m)
        ax.loglog(group["M"], group["runtime"], color=colors.get(m, colors.get(lbl, "black")),
                  marker=markers.get(m, markers.get(lbl, "o")), label=lbl)
    ax.set_xlabel("Radial Grid Points (M)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"N = {N_min} — Runtime vs M")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Top-right: runtime vs M for N = N_max
    ax = axes[0, 1]
    df_sub = df_results[df_results["N"] == N_max]
    for m, group in df_sub.groupby(col):
        group = group.sort_values("M")
        lbl = labels.get(m, m)
        ax.loglog(group["M"], group["runtime"], color=colors.get(m, colors.get(lbl, "black")),
                  marker=markers.get(m, markers.get(lbl, "o")), label=lbl)
    ax.set_xlabel("Radial Grid Points (M)")
    ax.set_title(f"N = {N_max} — Runtime vs M")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Bottom-left: runtime vs N for M = M_min
    ax = axes[1, 0]
    df_sub = df_results[df_results["M"] == M_min]
    for m, group in df_sub.groupby(col):
        group = group.sort_values("N")
        lbl = labels.get(m, m)
        ax.loglog(group["N"], group["runtime"], color=colors.get(m, colors.get(lbl, "black")),
                  marker=markers.get(m, markers.get(lbl, "s")), label=lbl)
    ax.set_xlabel("Angular Grid Points (N)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"M = {M_min} — Runtime vs N")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Bottom-right: runtime vs N for M = M_max
    ax = axes[1, 1]
    df_sub = df_results[df_results["M"] == M_max]
    for m, group in df_sub.groupby(col):
        group = group.sort_values("N")
        lbl = labels.get(m, m)
        ax.loglog(group["N"], group["runtime"], color=colors.get(m, colors.get(lbl, "black")),
                  marker=markers.get(m, markers.get(lbl, "s")), label=lbl)
    ax.set_xlabel("Angular Grid Points (N)")
    ax.set_title(f"M = {M_max} — Runtime vs N")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Single shared legend below the 2x2 plots
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, legend_labels = axes[1, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_labels) if legend_labels else 1,
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()


# Alias for consistent naming
plot_runtime_table1_extremes = plot_extreme_runtime_2x2