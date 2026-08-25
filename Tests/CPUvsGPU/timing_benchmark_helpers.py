import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display
from tqdm.auto import tqdm

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_stratified_rand_azimuthal,
    generate_jittered_azimuthal,
    generate_sine_perturbed_azimuthal,
    generate_chebyshev_angular_azimuthal,
    generate_cartesian_grid_on_disk,
    compute_zero_mode,
)
from Poisson_Solver.visualization import compute_error_metrics
from Poisson_Solver.poisson_solver import poisson_solver


# ==============================================================================
# Multi-Run Timing Configuration
# ==============================================================================
TIME_TRIALS = False  # Set to True to run each solve multiple times and record min runtime
NUM_RUNS = 5

def set_timing_config(time_trials=False, num_runs=5):
    """Globally configure multi-trial benchmark timing."""
    global TIME_TRIALS, NUM_RUNS
    TIME_TRIALS = bool(time_trials)
    NUM_RUNS = int(num_runs) if time_trials else 1


# ==============================================================================
# Manufactured Problem Definition
# ==============================================================================
def get_benchmark_problem(R=1.0, k=2):
    """
    Smooth manufactured nonseparable Dirichlet problem on the unit disk:
        u(x, y) = (R^2 - x^2 - y^2) * exp(x) * sin(k * y)
        u(R, theta) = 0 (Homogeneous Dirichlet)

    Analytical Laplacian:
        Δu = exp(x) * [
            (1 - k^2) * (R^2 - x^2 - y^2) * sin(k y)
            - 4 * (1 + x) * sin(k y)
            - 4 * k * y * cos(k y)
        ]
    """
    def u(x, y):
        r2 = x**2 + y**2
        return (R**2 - r2) * np.exp(x) * np.sin(k * y)

    def f(x, y):
        r2 = x**2 + y**2
        exp_x = np.exp(x)
        sin_ky = np.sin(k * y)
        cos_ky = np.cos(k * y)
        term1 = (1.0 - k**2) * (R**2 - r2) * sin_ky
        term2 = -4.0 * (1.0 + x) * sin_ky
        term3 = -4.0 * k * y * cos_ky
        return exp_x * (term1 + term2 + term3)

    def g_dir(x, y):
        return np.zeros_like(x)

    def g_neu(x, y):
        raise NotImplementedError("This benchmark is configured for Dirichlet conditions.")

    return {
        "name": "Cartesian Mixed (Dirichlet)",
        "u": u,
        "f": f,
        "g_dirichlet": g_dir,
        "g_neumann": g_neu,
        "R": R,
        "k": k,
    }


# ==============================================================================
# Grid Generation Utilities
# ==============================================================================
def generate_benchmark_azimuthal_grid(grid_name, N, **kwargs):
    """
    Generate an azimuthal angle grid theta_j in [0, 2*pi).

    Supported grid names:
      - 'uniform': Uniform equispaced
      - 'jittered': Jittered grid (Toeplitz Mesh, default jitter_fraction=0.25)
      - 'stratified_rand': Stratified random
      - 'sine_perturbed': Sine-perturbed smooth grid (CG Mesh, default amplitude=0.20, mode=2)
      - 'multipole': Multipole harmonic grid (default poles=(2,4), amps=(0.04,0.02))
      - 'warped': Warped conformal grid (default eps1=0.08, eps2=-0.04)
    """
    name_lower = grid_name.lower().strip()
    if name_lower in ("uniform", "unif"):
        return generate_uniform_azimuthal(N)
    elif name_lower in ("jittered", "jitter", "toeplitz_mesh"):
        jf = kwargs.get("jitter_fraction", 0.25)
        return generate_jittered_azimuthal(N, jitter_fraction=jf)
    elif name_lower in ("stratified_rand", "stratified_random", "stratified"):
        return generate_stratified_rand_azimuthal(N)
    elif name_lower in ("sine_perturbed", "sine", "mild_sine", "cg_mesh"):
        amp = kwargs.get("amplitude", 0.20)
        mode = kwargs.get("mode", 2)
        return generate_sine_perturbed_azimuthal(N, amplitude=amp, mode=mode)
    elif name_lower in ("multipole", "harmonic"):
        poles = kwargs.get("poles", (2, 4))
        amps = kwargs.get("amps", (0.04, 0.02))
        from Poisson_Solver.grids import generate_multipole_azimuthal
        return generate_multipole_azimuthal(N, poles=poles, amps=amps)
    elif name_lower in ("warped", "conformal"):
        eps1 = kwargs.get("eps1", 0.08)
        eps2 = kwargs.get("eps2", -0.04)
        from Poisson_Solver.grids import generate_warped_azimuthal
        return generate_warped_azimuthal(N, eps1=eps1, eps2=eps2)
    elif name_lower in ("chebyshev", "chebyshev_angular"):
        alpha = kwargs.get("alpha", 0.28)
        return generate_chebyshev_angular_azimuthal(N, alpha=alpha)
    else:
        raise ValueError(f"Unknown azimuthal grid name: '{grid_name}'")


# ==============================================================================
# Timing & Benchmark Execution Harness
# ==============================================================================
def timed_poisson_solve(f_vals, g_vals, u_fourier_0, N, M, r_m, theta_j, R,
                        quad_rule, BC_choice, rad_unif, grid_type,
                        use_nudft_angular=False, maxiter_nufft=50, tol_nufft=1e-8,
                        reg_param=1e-12, eps_finufft=1e-12, num_processors=None,
                        use_gpu=False, **kwargs):
    """
    Executes poisson_solver with multi-trial precision timing if TIME_TRIALS=True.
    Returns: (u_approx, elapsed_sec, min_sec, mean_sec)
    """
    solve_kwargs = dict(
        f_values=f_vals,
        g_values=g_vals,
        u_fourier_0=u_fourier_0,
        N=N,
        M=M,
        r_m=r_m,
        theta_j=theta_j,
        R=R,
        quad_rule=quad_rule,
        BC_choice=BC_choice,
        rad_unif=rad_unif,
        grid_type=grid_type,
        use_nudft_angular=use_nudft_angular,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps_finufft=eps_finufft,
        num_processors=num_processors,
        use_gpu=use_gpu,
        **kwargs
    )

    n_runs = NUM_RUNS if TIME_TRIALS else 1
    runtimes = []
    u_approx = None

    for i in range(n_runs):
        if use_gpu:
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
        t0 = time.perf_counter()
        res = poisson_solver(**solve_kwargs)
        if use_gpu:
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
        t1 = time.perf_counter()
        runtimes.append(t1 - t0)
        if u_approx is None:
            u_approx = res

    elapsed = float(np.min(runtimes)) if TIME_TRIALS else float(runtimes[0])
    min_time = float(np.min(runtimes))
    mean_time = float(np.mean(runtimes))
    return u_approx, elapsed, min_time, mean_time


# ==============================================================================
# Suite 1: Uniform Grid Benchmark
# ==============================================================================
def run_uniform_benchmark(N_list, M_list, problem=None, R=1.0, quad_rule=1,
                          BC_choice=1, perimeter_only=False, num_processors=None, use_gpu=False):
    """
    Benchmark Uniform FFT Poisson Solver across a 2D (N x M) grid matrix.
    """
    if problem is None:
        problem = get_benchmark_problem(R=R)

    u_exact_func = problem["u"]
    f_func = problem["f"]
    g_func = problem["g_dirichlet"]

    min_N, max_N = min(N_list), max(N_list)
    min_M, max_M = min(M_list), max(M_list)

    records = []
    tasks = []
    for N in N_list:
        for M in M_list:
            if perimeter_only and (N not in (min_N, max_N)) and (M not in (min_M, max_M)):
                continue
            tasks.append((N, M))

    pbar = tqdm(total=len(tasks), desc="Uniform FFT")

    for N, M in tasks:
        theta_j = generate_uniform_azimuthal(N)
        r_m = generate_uniform_radial(M, R)
        x_mesh, y_mesh = generate_cartesian_grid_on_disk(theta_j, r_m)

        f_vals = f_func(x_mesh, y_mesh)
        g_vals = g_func(x_mesh[:, -1], y_mesh[:, -1])
        u_exact = u_exact_func(x_mesh, y_mesh)
        u_fourier_0 = np.array([])

        u_approx, elapsed, min_t, mean_t = timed_poisson_solve(
            f_vals=f_vals,
            g_vals=g_vals,
            u_fourier_0=u_fourier_0,
            N=N,
            M=M,
            r_m=r_m,
            theta_j=theta_j,
            R=R,
            quad_rule=quad_rule,
            BC_choice=BC_choice,
            rad_unif=1,
            grid_type=1,
            num_processors=num_processors,
            use_gpu=use_gpu,
        )

        linf, linf_rel, l2, l2_rel = compute_error_metrics(u_approx, u_exact, r_m, theta_j)

        records.append({
            "N": N,
            "M": M,
            "Total_Points": N * M,
            "Grid_Category": "Uniform Grid",
            "Grid_Type": "uniform",
            "Solver": "Uniform FFT",
            "Time_sec": elapsed,
            "Time_ms": elapsed * 1000.0,
            "Min_Time_sec": min_t,
            "Mean_Time_sec": mean_t,
            "L_inf_Error": linf,
            "L2_Error": l2,
            "Rel_L2_Error": l2_rel,
            "Backend": "GPU" if use_gpu else "CPU",
        })
        pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# ==============================================================================
# Unified 5-Method Benchmark Runner
# ==============================================================================
def run_all_benchmarks(N_list, M_list, problem=None, R=1.0, quad_rule=1,
                       BC_choice=1, maxiter_nufft=100, tol_nufft=1e-8,
                       reg_param=1e-10, eps_finufft=1e-12,
                       perimeter_only=False,
                       toeplitz_grid="jittered", toeplitz_kwargs=None,
                       cg_grid="sine_perturbed", cg_kwargs=None,
                       num_processors=None, use_gpu=False):
    """
    Executes the unified 5-method benchmark across (N x M):
      1. Uniform FFT on Uniform Grid
      2. NUDFT on Toeplitz Nonuniform Mesh (default Jittered, delta=0.25)
      3. NUFFT Toeplitz on Toeplitz Nonuniform Mesh
      4. NUDFT on CG Nonuniform Mesh (default Sine-Perturbed, amplitude=0.20, mode=2)
      5. NUFFT CG (PCGLS) on CG Nonuniform Mesh
    """
    if problem is None:
        problem = get_benchmark_problem(R=R)

    if toeplitz_kwargs is None:
        toeplitz_kwargs = {"jitter_fraction": 0.25} if "jitter" in toeplitz_grid else {}
    if cg_kwargs is None:
        cg_kwargs = {"amplitude": 0.20, "mode": 2} if "sine" in cg_grid else {}

    u_exact_func = problem["u"]
    f_func = problem["f"]
    g_func = problem["g_dirichlet"]

    min_N, max_N = min(N_list), max(N_list)
    min_M, max_M = min(M_list), max(M_list)

    tasks = []
    for N in N_list:
        for M in M_list:
            if perimeter_only and (N not in (min_N, max_N)) and (M not in (min_M, max_M)):
                continue
            tasks.append((N, M))

    records = []
    total_solves = len(tasks) * 5
    pbar = tqdm(total=total_solves, desc=f"Benchmarking 5 Solvers [{'GPU' if use_gpu else 'CPU'}]")

    for N, M in tasks:
        r_m = generate_uniform_radial(M, R)
        u_fourier_0_dir = np.array([])
        u_fourier_0_nu = 0.0

        # -------------------------------------------------------------
        # 1. Uniform Grid -> Uniform FFT
        # -------------------------------------------------------------
        th_unif = generate_uniform_azimuthal(N)
        xu, yu = generate_cartesian_grid_on_disk(th_unif, r_m)
        fu = f_func(xu, yu)
        gu = g_func(xu[:, -1], yu[:, -1])
        uex_u = u_exact_func(xu, yu)

        u_ufft, t_ufft, min_ufft, mean_ufft = timed_poisson_solve(
            f_vals=fu, g_vals=gu, u_fourier_0=u_fourier_0_dir,
            N=N, M=M, r_m=r_m, theta_j=th_unif, R=R,
            quad_rule=quad_rule, BC_choice=BC_choice, rad_unif=1, grid_type=1,
            num_processors=num_processors, use_gpu=use_gpu
        )
        linf_ufft, _, l2_ufft, rel_l2_ufft = compute_error_metrics(u_ufft, uex_u, r_m, th_unif)
        records.append({
            "N": N, "M": M, "Total_Points": N * M,
            "Grid_Category": "Uniform Grid", "Grid_Type": "uniform",
            "Solver": "Uniform FFT",
            "Time_sec": t_ufft, "Time_ms": t_ufft * 1000.0,
            "Min_Time_sec": min_ufft, "Mean_Time_sec": mean_ufft,
            "L_inf_Error": linf_ufft, "L2_Error": l2_ufft, "Rel_L2_Error": rel_l2_ufft,
            "Backend": "GPU" if use_gpu else "CPU",
        })
        pbar.update(1)

        # -------------------------------------------------------------
        # 2 & 3. Toeplitz Mesh -> NUDFT & NUFFT Toeplitz
        # -------------------------------------------------------------
        th_toep = generate_benchmark_azimuthal_grid(toeplitz_grid, N, **toeplitz_kwargs)
        xt, yt = generate_cartesian_grid_on_disk(th_toep, r_m)
        ft = f_func(xt, yt)
        gt = g_func(xt[:, -1], yt[:, -1])
        uex_t = u_exact_func(xt, yt)

        # (a) NUDFT on Toeplitz Mesh
        u_nd_t, t_nd_t, min_nd_t, mean_nd_t = timed_poisson_solve(
            f_vals=ft, g_vals=gt, u_fourier_0=u_fourier_0_nu,
            N=N, M=M, r_m=r_m, theta_j=th_toep, R=R,
            quad_rule=quad_rule, BC_choice=BC_choice, rad_unif=1, grid_type=2,
            use_nudft_angular=True, reg_param=reg_param, eps_finufft=eps_finufft,
            num_processors=num_processors, use_gpu=use_gpu
        )
        linf_nd_t, _, l2_nd_t, rel_l2_nd_t = compute_error_metrics(u_nd_t, uex_t, r_m, th_toep)
        records.append({
            "N": N, "M": M, "Total_Points": N * M,
            "Grid_Category": "Toeplitz Mesh", "Grid_Type": toeplitz_grid,
            "Solver": "NUDFT (Toeplitz Mesh)",
            "Time_sec": t_nd_t, "Time_ms": t_nd_t * 1000.0,
            "Min_Time_sec": min_nd_t, "Mean_Time_sec": mean_nd_t,
            "L_inf_Error": linf_nd_t, "L2_Error": l2_nd_t, "Rel_L2_Error": rel_l2_nd_t,
            "Backend": "GPU" if use_gpu else "CPU",
        })
        pbar.update(1)

        # (b) NUFFT Toeplitz on Toeplitz Mesh
        u_nf_t, t_nf_t, min_nf_t, mean_nf_t = timed_poisson_solve(
            f_vals=ft, g_vals=gt, u_fourier_0=u_fourier_0_nu,
            N=N, M=M, r_m=r_m, theta_j=th_toep, R=R,
            quad_rule=quad_rule, BC_choice=BC_choice, rad_unif=1, grid_type=2,
            use_nudft_angular=False, maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft,
            reg_param=reg_param, eps_finufft=eps_finufft,
            num_processors=num_processors, use_gpu=use_gpu
        )
        linf_nf_t, _, l2_nf_t, rel_l2_nf_t = compute_error_metrics(u_nf_t, uex_t, r_m, th_toep)
        records.append({
            "N": N, "M": M, "Total_Points": N * M,
            "Grid_Category": "Toeplitz Mesh", "Grid_Type": toeplitz_grid,
            "Solver": "NUFFT Toeplitz",
            "Time_sec": t_nf_t, "Time_ms": t_nf_t * 1000.0,
            "Min_Time_sec": min_nf_t, "Mean_Time_sec": mean_nf_t,
            "L_inf_Error": linf_nf_t, "L2_Error": l2_nf_t, "Rel_L2_Error": rel_l2_nf_t,
            "Backend": "GPU" if use_gpu else "CPU",
        })
        pbar.update(1)

        # -------------------------------------------------------------
        # 4 & 5. CG Mesh -> NUDFT & NUFFT CG (PCGLS)
        # -------------------------------------------------------------
        th_cg = generate_benchmark_azimuthal_grid(cg_grid, N, **cg_kwargs)
        xc, yc = generate_cartesian_grid_on_disk(th_cg, r_m)
        fc = f_func(xc, yc)
        gc = g_func(xc[:, -1], yc[:, -1])
        uex_c = u_exact_func(xc, yc)

        # (a) NUDFT on CG Mesh
        u_nd_c, t_nd_c, min_nd_c, mean_nd_c = timed_poisson_solve(
            f_vals=fc, g_vals=gc, u_fourier_0=u_fourier_0_nu,
            N=N, M=M, r_m=r_m, theta_j=th_cg, R=R,
            quad_rule=quad_rule, BC_choice=BC_choice, rad_unif=1, grid_type=3,
            use_nudft_angular=True, reg_param=reg_param, eps_finufft=eps_finufft,
            num_processors=num_processors, use_gpu=use_gpu
        )
        linf_nd_c, _, l2_nd_c, rel_l2_nd_c = compute_error_metrics(u_nd_c, uex_c, r_m, th_cg)
        records.append({
            "N": N, "M": M, "Total_Points": N * M,
            "Grid_Category": "CG Mesh", "Grid_Type": cg_grid,
            "Solver": "NUDFT (CG Mesh)",
            "Time_sec": t_nd_c, "Time_ms": t_nd_c * 1000.0,
            "Min_Time_sec": min_nd_c, "Mean_Time_sec": mean_nd_c,
            "L_inf_Error": linf_nd_c, "L2_Error": l2_nd_c, "Rel_L2_Error": rel_l2_nd_c,
            "Backend": "GPU" if use_gpu else "CPU",
        })
        pbar.update(1)

        # (b) NUFFT CG (PCGLS) on CG Mesh
        u_nf_c, t_nf_c, min_nf_c, mean_nf_c = timed_poisson_solve(
            f_vals=fc, g_vals=gc, u_fourier_0=u_fourier_0_nu,
            N=N, M=M, r_m=r_m, theta_j=th_cg, R=R,
            quad_rule=quad_rule, BC_choice=BC_choice, rad_unif=1, grid_type=3,
            use_nudft_angular=False, maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft,
            reg_param=reg_param, eps_finufft=eps_finufft,
            num_processors=num_processors, use_gpu=use_gpu
        )
        linf_nf_c, _, l2_nf_c, rel_l2_nf_c = compute_error_metrics(u_nf_c, uex_c, r_m, th_cg)
        records.append({
            "N": N, "M": M, "Total_Points": N * M,
            "Grid_Category": "CG Mesh", "Grid_Type": cg_grid,
            "Solver": "NUFFT CG (PCGLS)",
            "Time_sec": t_nf_c, "Time_ms": t_nf_c * 1000.0,
            "Min_Time_sec": min_nf_c, "Mean_Time_sec": mean_nf_c,
            "L_inf_Error": linf_nf_c, "L2_Error": l2_nf_c, "Rel_L2_Error": rel_l2_nf_c,
            "Backend": "GPU" if use_gpu else "CPU",
        })
        pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# ==============================================================================
# Matrix Table Formatters
# ==============================================================================
def display_benchmark_tables(df, N_values=None, M_values=None, title_prefix=""):
    """
    Display comprehensive N x M timing and accuracy tables across all 5 methods.
    """
    solver_order = [
        "Uniform FFT",
        "NUDFT (Toeplitz Mesh)",
        "NUFFT Toeplitz",
        "NUDFT (CG Mesh)",
        "NUFFT CG (PCGLS)",
    ]
    solvers = [s for s in solver_order if s in df["Solver"].values]
    if not solvers:
        solvers = list(df["Solver"].unique())

    backend_str = f" [{df['Backend'].iloc[0]}]" if "Backend" in df.columns else ""

    def fmt_time(x):
        return "—" if pd.isna(x) else f"{x:.4f} s"

    def fmt_err(x):
        return "—" if pd.isna(x) else f"{x:.2e}"

    # 1. Timing Matrix Table
    print(f"\n{'='*90}\n{title_prefix}TIMING MATRIX (s){backend_str}\n{'='*90}")
    pivots_time = {}
    for s in solvers:
        p = df[df["Solver"] == s].pivot(index="N", columns="M", values="Time_sec")
        if N_values is not None and M_values is not None:
            p = p.reindex(index=N_values, columns=M_values)
        pivots_time[s] = p
    table_time = pd.concat(pivots_time, axis=1)
    display(table_time.map(fmt_time))

    # 2. Accuracy Matrix Table
    print(f"\n{'='*90}\n{title_prefix}ACCURACY MATRIX (L_inf Error){backend_str}\n{'='*90}")
    pivots_acc = {}
    for s in solvers:
        p = df[df["Solver"] == s].pivot(index="N", columns="M", values="L_inf_Error")
        if N_values is not None and M_values is not None:
            p = p.reindex(index=N_values, columns=M_values)
        pivots_acc[s] = p
    table_acc = pd.concat(pivots_acc, axis=1)
    display(table_acc.map(fmt_err))


# ==============================================================================
# Unified 2x2 Log-Log Visualization Functions
# ==============================================================================
def plot_grid_distributions(N=64, R=1.0):
    """
    Plot polar scatter representations of the tested angular meshes.
    """
    grid_types = [
        ("Uniform Grid (Uniform FFT)", generate_benchmark_azimuthal_grid("uniform", N)),
        ("Jittered Grid (Toeplitz, delta=0.25)", generate_benchmark_azimuthal_grid("jittered", N, jitter_fraction=0.25)),
        ("Sine-Perturbed Grid (CG, a=0.20)", generate_benchmark_azimuthal_grid("sine_perturbed", N, amplitude=0.20, mode=2)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), subplot_kw={'projection': 'polar'})
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for ax, (title, thetas), color in zip(axes, grid_types, colors):
        r_circ = np.ones_like(thetas) * R
        ax.scatter(thetas, r_circ, color=color, s=32, alpha=0.85, edgecolors='black', linewidth=0.5, zorder=3)
        ax.plot(np.linspace(0, 2*np.pi, 200), np.ones(200)*R, 'k--', alpha=0.4, linewidth=1.0)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        ax.set_yticklabels([])
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Angular Mesh Distributions on Unit Circle (N = {N})", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig


def plot_runtime_2x2(df, N_values=None, M_values=None, use_gpu=None, title="Runtime Scaling Analysis"):
    """
    Unified 2x2 Log-Log Runtime Figure (in Seconds) plotting all 5 solvers.
      (0, 0): Runtime vs N for min(M)
      (0, 1): Runtime vs N for max(M)
      (1, 0): Runtime vs M for min(N)
      (1, 1): Runtime vs M for max(N)
    """
    if N_values is None:
        N_values = sorted(df["N"].unique())
    if M_values is None:
        M_values = sorted(df["M"].unique())

    if use_gpu is None:
        use_gpu = ("Backend" in df.columns and (df["Backend"] == "GPU").any())

    backend_tag = "[GPU]" if use_gpu else "[CPU]"
    fig_title = f"{title} {backend_tag}"

    min_M, max_M = min(M_values), max(M_values)
    min_N, max_N = min(N_values), max(N_values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    solver_styles = {
        "Uniform FFT": ("#1f77b4", "o", "-"),
        "NUDFT (Toeplitz Mesh)": ("#2ca02c", "s", "--"),
        "NUFFT Toeplitz": ("#17becf", "s", "-"),
        "NUDFT (CG Mesh)": ("#d62728", "^", "--"),
        "NUFFT CG (PCGLS)": ("#ff7f0e", "^", "-"),
    }

    configs = [
        (axes[0, 0], "N", "Time_sec", df[df["M"] == min_M], f"(a) Runtime vs. N  (Fixed M = {min_M})", "Azimuthal Resolution N"),
        (axes[0, 1], "N", "Time_sec", df[df["M"] == max_M], f"(b) Runtime vs. N  (Fixed M = {max_M})", "Azimuthal Resolution N"),
        (axes[1, 0], "M", "Time_sec", df[df["N"] == min_N], f"(c) Runtime vs. M  (Fixed N = {min_N})", "Radial Resolution M"),
        (axes[1, 1], "M", "Time_sec", df[df["N"] == max_N], f"(d) Runtime vs. M  (Fixed N = {max_N})", "Radial Resolution M"),
    ]

    for ax, x_col, y_col, sub_df, sub_title, x_label in configs:
        for solver_name, (color, marker, ls) in solver_styles.items():
            grp = sub_df[sub_df["Solver"] == solver_name]
            if grp.empty:
                continue
            grp_sorted = grp.dropna(subset=[y_col]).sort_values(x_col)
            if grp_sorted.empty:
                continue
            ax.loglog(grp_sorted[x_col], grp_sorted[y_col], marker=marker, linestyle=ls,
                      linewidth=2.0, markersize=7, label=solver_name, color=color)

        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel("Execution Time (s)", fontsize=10, fontweight='bold')
        ax.set_title(sub_title, fontsize=11, fontweight='bold')
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8.5, loc="upper left", frameon=True)

    plt.suptitle(fig_title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_accuracy_2x2(df, N_values=None, M_values=None, metric="L_inf_Error", use_gpu=None, title="Accuracy Scaling Analysis"):
    """
    Unified 2x2 Log-Log Accuracy Figure plotting all 5 solvers.
      (0, 0): L_inf Error vs N for min(M)
      (0, 1): L_inf Error vs N for max(M)
      (1, 0): L_inf Error vs M for min(N)
      (1, 1): L_inf Error vs M for max(N)
    """
    if N_values is None:
        N_values = sorted(df["N"].unique())
    if M_values is None:
        M_values = sorted(df["M"].unique())

    if use_gpu is None:
        use_gpu = ("Backend" in df.columns and (df["Backend"] == "GPU").any())

    backend_tag = "[GPU]" if use_gpu else "[CPU]"
    fig_title = f"{title} {backend_tag}"

    min_M, max_M = min(M_values), max(M_values)
    min_N, max_N = min(N_values), max(N_values)

    metric_name = r"$L_\infty$ Error" if "inf" in metric.lower() else r"$L_2$ Error"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    solver_styles = {
        "Uniform FFT": ("#1f77b4", "o", "-"),
        "NUDFT (Toeplitz Mesh)": ("#2ca02c", "s", "--"),
        "NUFFT Toeplitz": ("#17becf", "s", "-"),
        "NUDFT (CG Mesh)": ("#d62728", "^", "--"),
        "NUFFT CG (PCGLS)": ("#ff7f0e", "^", "-"),
    }

    configs = [
        (axes[0, 0], "N", metric, df[df["M"] == min_M], f"(a) Error vs. N  (Fixed M = {min_M})", "Azimuthal Resolution N"),
        (axes[0, 1], "N", metric, df[df["M"] == max_M], f"(b) Error vs. N  (Fixed M = {max_M})", "Azimuthal Resolution N"),
        (axes[1, 0], "M", metric, df[df["N"] == min_N], f"(c) Error vs. M  (Fixed N = {min_N})", "Radial Resolution M"),
        (axes[1, 1], "M", metric, df[df["N"] == max_N], f"(d) Error vs. M  (Fixed N = {max_N})", "Radial Resolution M"),
    ]

    for ax, x_col, y_col, sub_df, sub_title, x_label in configs:
        for solver_name, (color, marker, ls) in solver_styles.items():
            grp = sub_df[sub_df["Solver"] == solver_name]
            if grp.empty:
                continue
            grp_sorted = grp.dropna(subset=[y_col]).sort_values(x_col)
            if grp_sorted.empty:
                continue
            ax.loglog(grp_sorted[x_col], grp_sorted[y_col], marker=marker, linestyle=ls,
                      linewidth=2.0, markersize=7, label=solver_name, color=color)

        ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(sub_title, fontsize=11, fontweight='bold')
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8.5, loc="lower left", frameon=True)

    plt.suptitle(fig_title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig
