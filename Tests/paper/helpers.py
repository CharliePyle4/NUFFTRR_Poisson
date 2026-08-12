import os
import sys
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
    generate_cartesian_grid_on_disk,
    compute_zero_mode,
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver


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


def generate_multipole_azimuthal(N, poles=(2, 4), amplitudes=(0.14, 0.08)):
    """Structured periodic distortion theta=xi+sum(a_p sin(p xi)); requires sum(|a_p|p)<1."""
    poles = np.asarray(poles, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if poles.ndim != 1 or amplitudes.ndim != 1 or len(poles) != len(amplitudes):
        raise ValueError("poles and amplitudes must be equal-length 1-D arrays")
    if np.sum(np.abs(poles) * np.abs(amplitudes)) >= 1.0:
        raise ValueError("Require sum(abs(poles)*abs(amplitudes)) < 1")
    xi = 2.0 * np.pi * (np.arange(N) + 0.5) / N
    theta = xi + sum(a * np.sin(p * xi) for p, a in zip(poles, amplitudes))
    return np.sort(np.mod(theta, 2.0 * np.pi))


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
    Periodic cubic-spline interpolation from distorted angular samples
    onto target angular positions.

    theta_src:
        Sorted distorted angular angles, shape (P,).

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
):
    """
    Solve the disk Poisson problem from structured distorted measurements.

    Timed work:
        - Uniform-grid interpolation, when azu_unif == 2.
        - Poisson solver execution.

    Untimed work:
        - Analytical forcing evaluation.
        - Exact solution evaluation.
        - Error metric calculation.
        - Plot-data preparation.
    """
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
    # Start timing:
    # interpolation/gridding plus Poisson solve only.
    # --------------------------------------------------------------
    t0 = time.perf_counter()

    if azu_unif == 2:
        # Uniform FFT pipeline:
        # distorted measurements -> periodic cubic-spline interpolation ->
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
        # consume distorted angular measurements directly.
        theta_solver = theta_raw
        f_values = f_raw
        g_values = g_raw

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
        x_bc, y_bc = generate_cartesian_grid_on_disk(
            theta_solver,
            r_m,
        )

        u_bc = problem["u"](
            x_bc,
            y_bc,
        )

        u_fourier_0 = compute_zero_mode(
            u_bc,
            theta_solver,
            azu_unif,
        )[-1]

    # --------------------------------------------------------------
    # Timed Poisson solve.
    # --------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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
            azu_unif=azu_unif,
            grid_type=(1 if azu_unif == 2 else 3),
            use_nudft_angular=use_nudft,
            maxiter_nufft=maxiter_nufft,
            tol_nufft=tol_nufft,
        )

    runtime = time.perf_counter() - t0

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


def run_all_algorithms_NxM_study(problem, N_values, M_values, poles=(2, 4),
                                  amplitudes=(0.14, 0.08), bc_choice=1, quad_rule=2):
    algorithms = [
        ("Adapted NUFFT", 1, False),
        ("Adapted NUDFT", 1, True),
        ("Uniform FFT + periodic cubic spline", 2, False),
    ]
    rows = []
    pbar = tqdm(total=len(N_values)*len(M_values)*len(algorithms),
                desc="Computing N x M multipole-grid study")
    for N in N_values:
        theta = generate_multipole_azimuthal(N, poles, amplitudes)
        for M in M_values:
            for name, azu_unif, use_nudft in algorithms:
                row = run_benchmark_case(N, M, azu_unif, theta, problem, bc_choice,
                                         quad_rule, use_nudft)
                row["method"] = name
                rows.append(row)
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def plot_solution_and_grids(problem, N_adapt=32, N_unif=32, M=32,
                            poles=(2, 4), amplitudes=(0.14, 0.08)):
    R = problem["R"]
    theta_adapt = generate_multipole_azimuthal(N_adapt, poles, amplitudes)
    theta_unif = generate_uniform_azimuthal(N_unif)
    fine_theta = np.linspace(0, 2*np.pi, 250, endpoint=False)
    fine_r = np.linspace(0, R, 120)
    Xf, Yf = generate_cartesian_grid_on_disk(fine_theta, fine_r)
    fig = plt.figure(figsize=(11, 3.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    surf = ax.plot_surface(Xf, Yf, problem["u"](Xf, Yf), cmap="plasma", edgecolor="none")
    ax.set_title("Exact Multipole Solution")
    fig.colorbar(surf, ax=ax, shrink=.5, aspect=10)

    ax = fig.add_subplot(1, 3, 2)
    for rr in np.linspace(.2*R, R, 5):
        ax.add_patch(Circle((0, 0), rr, fill=False, ec=".82", ls="--", lw=.8))
    for rr in (.4*R, .7*R, R):
        ax.scatter(rr*np.cos(theta_unif), rr*np.sin(theta_unif), s=16, c="royalblue",
                   alpha=.6, label=f"Uniform FFT target grid (N={N_unif})" if rr == R else None)
        ax.scatter(rr*np.cos(theta_adapt), rr*np.sin(theta_adapt), s=24, c="crimson", marker="^",
                   alpha=.85, label=f"Structured distorted grid (N={N_adapt})" if rr == R else None)
    ax.set_aspect("equal"); ax.set_xlim(-1.15*R, 1.15*R); ax.set_ylim(-1.15*R, 1.15*R)
    ax.set_title("Structured Angular Deformation"); ax.legend(fontsize=8); ax.axis("off")

    ax = fig.add_subplot(1, 3, 3); rr = .7*R
    value = lambda th: problem["u"](rr*np.cos(th), rr*np.sin(th))
    ax.plot(fine_theta, value(fine_theta), "k-", lw=2, label="Exact angular profile")
    ax.scatter(theta_unif, value(theta_unif), c="royalblue", s=28, label="Uniform target nodes")
    ax.scatter(theta_adapt, value(theta_adapt), c="crimson", marker="^", s=36,
               label="Distorted measurement nodes")
    ax.set_title("Multipole Mode and Node Locations"); ax.set_xlabel(r"$\theta$"); ax.grid(alpha=.4); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


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
    poles=(2, 4),
    amplitudes=(0.08, 0.04),
    quad_rule=2,
    bc_choice=1,
):
    """
    Render a 2 x 2 disk-error comparison.

    Top-left:
        Adapted NUFFT using N_adapt distorted measurements.

    Top-right:
        Adapted NUDFT using the same N_adapt distorted measurements.

    Bottom-left:
        Uniform FFT + periodic cubic spline using the same N_adapt
        distorted measurements.

    Bottom-right:
        Uniform FFT + periodic cubic spline using N_unif_high
        distorted measurements.

    The first three cases have the same angular measurement budget.
    The final case is a higher-data-budget Uniform FFT comparison.
    """
    # --------------------------------------------------------------
    # Low-resolution distorted measurement grid:
    # shared by NUFFT, NUDFT, and equal-budget Uniform FFT.
    # --------------------------------------------------------------
    theta_low = generate_multipole_azimuthal(
        N_adapt,
        poles=poles,
        amplitudes=amplitudes,
    )

    # --------------------------------------------------------------
    # High-resolution distorted measurement grid:
    # used only by the final Uniform FFT comparison.
    # --------------------------------------------------------------
    theta_high = generate_multipole_azimuthal(
        N_unif_high,
        poles=poles,
        amplitudes=amplitudes,
    )

    # --------------------------------------------------------------
    # Case 1: NUFFT directly uses N_adapt distorted samples.
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
    )

    # --------------------------------------------------------------
    # Case 2: NUDFT directly uses the same distorted samples.
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
    )

    # --------------------------------------------------------------
    # Case 3: Uniform FFT gets the same low-N distorted samples,
    # then linearly interpolates them onto its uniform target grid.
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
    )

    # --------------------------------------------------------------
    # Case 4: Uniform FFT receives more distorted measurements,
    # then linearly interpolates them to a higher-resolution uniform grid.
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
    )

    cases = [
        (
            "Adapted NUFFT — direct distorted data",
            res_nufft,
        ),
        (
            "Adapted NUDFT — direct distorted data",
            res_nudft,
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
    # Print a concise numerical table for all four cases.
    # --------------------------------------------------------------
    summary_rows = []

    for label, res in cases:
        summary_rows.append(
            {
                "Case": label.replace("\n", " "),
                "N": res["N"],
                "M": res["M"],
                "Relative L2": res["L2_rel"],
                "Relative Linf": res["Linf_rel"],
                "Runtime (s)": res["runtime"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 95)
    print("2 x 2 Disk Error Comparison")
    print("=" * 95)

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
    # Use one common vertical error range across all four surfaces.
    # This makes the 3D comparison visually honest.
    # --------------------------------------------------------------
    all_errors = [
        np.abs(res["u_true"] - res["u_approx"])
        for _, res in cases
    ]

    error_max = max(
        np.max(error)
        for error in all_errors
    )

    if error_max <= 0:
        error_max = 1.0

    # --------------------------------------------------------------
    # Render 2 x 2 error surfaces.
    # --------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8.0))

    for plot_index, ((label, res), error) in enumerate(
        zip(cases, all_errors),
        start=1,
    ):
        ax = fig.add_subplot(
            2,
            2,
            plot_index,
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
            vmin=0.0,
            vmax=error_max,
        )

        ax.set_zlim(0.0, error_max)

        ax.set_title(
            f"{label}\n"
            f"N={res['N']}, M={res['M']}\n"
            f"$L_2$ = {res['L2_rel']:.2e} | "
            f"$L_\\infty$ = {res['Linf_rel']:.2e}",
            fontsize=9,
        )

        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("Pointwise error", fontsize=8)

        fig.colorbar(
            surface,
            ax=ax,
            shrink=0.5,
            aspect=10,
            pad=0.08,
        )

    fig.suptitle(
        "Equal-Budget Direct Nonuniform Solves vs. "
        "Uniform FFT Interpolation Pipelines",
        fontsize=13,
        y=0.98,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

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


def plot_accuracy_from_df(df_results, fixed_M=64, fixed_N=64):
    colors = {"Adapted NUFFT":"crimson", "Adapted NUDFT":"forestgreen", "Uniform FFT + periodic cubic spline":"royalblue"}
    markers = {"Adapted NUFFT":"s--", "Adapted NUDFT":"^-.", "Uniform FFT + periodic cubic spline":"o-"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for method, g in df_results[df_results.M == fixed_M].groupby("method"):
        g = g.sort_values("N"); axes[0].loglog(g.N, g.L2_rel, markers[method], color=colors[method], label=method)
    for method, g in df_results[df_results.N == fixed_N].groupby("method"):
        g = g.sort_values("M"); axes[1].loglog(g.M, g.L2_rel, markers[method], color=colors[method], label=method)
    axes[0].set(title=f"Error vs N (M={fixed_M})", xlabel="Angular points N", ylabel=r"Relative $L_2$ error")
    axes[1].set(title=f"Error vs M (N={fixed_N})", xlabel="Radial points M")
    for ax in axes: ax.grid(True, which="both", ls="--", alpha=.45); ax.legend(fontsize=7)
    fig.suptitle("Single Multipole on Structured Distorted Angular Grid — Accuracy", y=1.02)
    plt.tight_layout(); plt.show()


def plot_accuracy_faceted_by_method(df_results, N_values=None, M_values=None):
    methods = [m for m in ["Adapted NUFFT", "Adapted NUDFT", "Uniform FFT + periodic cubic spline"] if m in set(df_results.method)]
    N_values = sorted(df_results.N.unique()) if N_values is None else N_values
    M_values = sorted(df_results.M.unique()) if M_values is None else M_values
    fig, axes = plt.subplots(2, len(methods), figsize=(3.5*len(methods), 5), sharey="row", squeeze=False)
    cmap = plt.get_cmap("tab10")
    for col, method in enumerate(methods):
        sub = df_results[df_results.method == method]
        for i, N in enumerate(N_values):
            g = sub[sub.N == N].sort_values("M")
            axes[0,col].loglog(g.M, g.L2_rel, "o-", ms=3, color=cmap(i), label=f"N={N}")
        for i, M in enumerate(M_values):
            g = sub[sub.M == M].sort_values("N")
            axes[1,col].loglog(g.N, g.L2_rel, "s-", ms=3, color=cmap(i), label=f"M={M}")
        axes[0,col].set(title=f"{method}\nError vs M", xlabel="Radial points M")
        axes[1,col].set(title=f"{method}\nError vs N", xlabel="Angular points N")
        for ax in (axes[0,col], axes[1,col]): ax.grid(True, which="both", ls="--", alpha=.45); ax.legend(fontsize=6)
    axes[0,0].set_ylabel(r"Relative $L_2$ error"); axes[1,0].set_ylabel(r"Relative $L_2$ error")
    fig.suptitle("Single Multipole on Structured Distorted Angular Grid — Accuracy by Method", y=.99)
    plt.tight_layout(rect=(0, 0, 1, .95)); plt.show()


def plot_extreme_runtime_2x2(df_results, N_min=None, N_max=None, M_min=None, M_max=None):
    N_min = df_results.N.min() if N_min is None else N_min; N_max = df_results.N.max() if N_max is None else N_max
    M_min = df_results.M.min() if M_min is None else M_min; M_max = df_results.M.max() if M_max is None else M_max
    colors = {"Adapted NUFFT":"crimson", "Adapted NUDFT":"forestgreen", "Uniform FFT + periodic cubic spline":"royalblue"}
    specs = [(df_results[df_results.N == N_min], "M", f"N={N_min}: Runtime vs M"),
             (df_results[df_results.N == N_max], "M", f"N={N_max}: Runtime vs M"),
             (df_results[df_results.M == M_min], "N", f"M={M_min}: Runtime vs N"),
             (df_results[df_results.M == M_max], "N", f"M={M_max}: Runtime vs N")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6)); axes = axes.ravel()
    for ax, (sub, x, title) in zip(axes, specs):
        for method, g in sub.groupby("method"):
            g = g.sort_values(x); ax.loglog(g[x], g.runtime, "o-", color=colors.get(method, "black"), label=method)
        ax.set(title=title, xlabel=f"{x} grid points", ylabel="Runtime (seconds)"); ax.grid(True, which="both", ls="--", alpha=.45)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Single Multipole on Structured Distorted Angular Grid — Runtime", y=.99)
    plt.tight_layout(rect=(0, 0, 1, .92)); plt.show()

