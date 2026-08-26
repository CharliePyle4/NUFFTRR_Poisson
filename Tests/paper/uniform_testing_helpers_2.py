from tqdm.asyncio import tqdm
import numpy as np
import pandas as pd
import sympy as sp
from IPython.display import display
import time

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

# ==============================================================================
# Multi-Run Timing Configuration & Global Backend
# ==============================================================================
TIME_TRIALS = False  # Set to True to run each solve 5 times and record min runtime
NUM_RUNS = 5
GLOBAL_USE_GPU = False

def set_timing_config(time_trials=False, num_runs=5, use_gpu=False):
    """Globally configure multi-trial benchmark timing and backend."""
    global TIME_TRIALS, NUM_RUNS, GLOBAL_USE_GPU
    TIME_TRIALS = bool(time_trials)
    NUM_RUNS = int(num_runs) if time_trials else 1
    GLOBAL_USE_GPU = bool(use_gpu)

ANGLE_MESH_CACHE = {}

def get_problem_functions(u_sym, x, y):
    # Compute required derivatives
    f_sym = sp.diff(u_sym, x, 2) + sp.diff(u_sym, y, 2)
    u_x_sym = sp.diff(u_sym, x)
    u_y_sym = sp.diff(u_sym, y)

    # Convert to callable numpy functions
    u = sp.lambdify((x, y), u_sym, "numpy")
    f = sp.lambdify((x, y), f_sym, "numpy")
    u_x = sp.lambdify((x, y), u_x_sym, "numpy")
    u_y = sp.lambdify((x, y), u_y_sym, "numpy")

    g_dirichlet = lambda x_val, y_val: u(x_val, y_val)
    g_neumann = lambda x_val, y_val, R_val: (u_x(x_val, y_val) * x_val + u_y(x_val, y_val) * y_val) / R_val

    return u, f, g_dirichlet, g_neumann

def problem_5_setup():
    x, y = sp.symbols('x y')
    # Test Problem 5: Highly oscillatory function
    u_sym = sp.sin(10 * sp.pi * x) * sp.cos(10 * sp.pi * y)
    return get_problem_functions(u_sym, x, y)

def problem_6_setup():
    x, y = sp.symbols('x y')
    # Test Problem 6: Sharp boundary layer / Gaussian peak
    u_sym = sp.exp(-50 * ((x - 0.5)**2 + y**2))
    return get_problem_functions(u_sym, x, y)

def setup_problem_7():
    x, y = sp.symbols('x y')
    # Test Problem 7: Off-center singularity (logarithmic)
    u_sym = sp.log((x - 1.1)**2 + (y + 1.1)**2)
    return get_problem_functions(u_sym, x, y)

def get_cached_angle_mesh(method_cfg, N, M):
    azu_unif = method_cfg["azu_unif"]
    mesh_kind = method_cfg["mesh_kind"]
    method_name = method_cfg["name"]

    if azu_unif == 2:
        return generate_uniform_azimuthal(N)

    if azu_unif == 1:
        key = (method_name, N, mesh_kind)
        if key not in ANGLE_MESH_CACHE:
            ANGLE_MESH_CACHE[key] = generate_fixed_nonuniform_azimuthal(
                N, kind=mesh_kind or "rand"
            )
        return ANGLE_MESH_CACHE[key]

    if azu_unif == 0:
        key = (method_name, N, M, mesh_kind)
        if key not in ANGLE_MESH_CACHE:
            ANGLE_MESH_CACHE[key] = generate_nonuniform_azimuthal(
                N, M, kind=mesh_kind or "rand"
            )
        return ANGLE_MESH_CACHE[key]

    raise ValueError("Incorrect index for 'azu_unif'")

def build_radial_mesh(M, rad_unif, R):
    if rad_unif == 1:
        return generate_uniform_radial(M, R)
    return generate_nonuniform_radial(M, R)

def run_single_case(N, M, method_cfg, bc_name, quad_name, u, f, g_dirichlet, g_neumann,
                    BC_MAP, QUAD_MAP, rad_unif, R, num_processors=None, use_gpu=None,
                    time_trials=None, num_runs=None, maxiter_nufft=50, tol_nufft=1e-8,
                    reg_param=1e-12, eps_finufft=1e-12, **kwargs):
    bc_choice = BC_MAP[bc_name]
    quad_rule = QUAD_MAP[quad_name]

    azu_unif = method_cfg["azu_unif"]
    use_nudft = method_cfg["use_nudft"]

    iRadius = build_radial_mesh(M, rad_unif, R)
    iAngle  = get_cached_angle_mesh(method_cfg, N, M)

    x_coord, y_coord = generate_cartesian_grid_on_disk(iAngle, iRadius)

    # interior data and true solution
    f_values = generate_grid_values(f, x_coord, y_coord)
    u_true   = generate_grid_values(u, x_coord, y_coord)

    # boundary data depends on BC
    if bc_choice == 1:  # Dirichlet
        g_values = generate_grid_values(
            g_dirichlet, x_coord[:, M - 1], y_coord[:, M - 1]
        )
    elif bc_choice == 2:  # Neumann
        g_values = generate_grid_values(
            lambda x_val, y_val: g_neumann(x_val, y_val, R), x_coord[:, M - 1], y_coord[:, M - 1]
        )
    else:
        raise ValueError("Unknown bc_choice")

    # n = 0 mode for Neumann (phi_0), empty for Dirichlet
    if bc_choice == 2:
        u_fourier_0_arr = compute_zero_mode(u_true, iAngle, method_cfg["azu_unif"])
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    n_runs = num_runs if num_runs is not None else (NUM_RUNS if (TIME_TRIALS if time_trials is None else bool(time_trials)) else 1)
    actual_use_gpu = GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)

    try:
        runtimes = []
        u_approx = None
        for _ in range(n_runs):
            if actual_use_gpu:
                try:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()
                except Exception:
                    pass

            start_time = time.perf_counter()

            u_approx = poisson_solver(
                f_values, g_values, u_fourier_0,
                N, M, iRadius, iAngle, R,
                quad_rule, bc_choice,
                rad_unif, azu_unif,
                use_nudft_angular=(use_nudft if use_nudft is not None else False),
                maxiter_nufft=maxiter_nufft,
                tol_nufft=tol_nufft,
                reg_param=reg_param,
                eps_finufft=eps_finufft,
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

            runtimes.append(time.perf_counter() - start_time)

        solve_time = min(runtimes)

        _, linf_rel, _, l2_rel = compute_error_metrics(
            u_approx, u_true, iRadius, iAngle
        )

    except MemoryError:
        linf_rel = np.nan
        l2_rel = np.nan
        solve_time = np.nan

    return {
        "method": method_cfg["name"],
        "label": method_cfg["label"],
        "N": N,
        "M": M,
        "bc": bc_name,
        "quad": quad_name,
        "L_inf_rel": linf_rel,
        "L2_rel": l2_rel,
        "time": solve_time,
    }

def solve_for_grids(N, M, method_cfg, bc_name, quad_name, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R,
                    num_processors=None, use_gpu=None, maxiter_nufft=50, tol_nufft=1e-8, reg_param=1e-12, eps_finufft=1e-12, **kwargs):
    bc_choice = BC_MAP[bc_name]
    quad_rule = QUAD_MAP[quad_name]

    azu_unif = method_cfg["azu_unif"]
    use_nudft = method_cfg.get("use_nudft", False)
    actual_use_gpu = GLOBAL_USE_GPU if use_gpu is None else bool(use_gpu)

    iRadius = build_radial_mesh(M, rad_unif, R)
    iAngle  = get_cached_angle_mesh(method_cfg, N, M)

    x_coord, y_coord = generate_cartesian_grid_on_disk(iAngle, iRadius)

    f_values = generate_grid_values(f, x_coord, y_coord)
    u_true   = generate_grid_values(u, x_coord, y_coord)

    if bc_choice == 1:
        g_values = generate_grid_values(g_dirichlet, x_coord[:, M - 1], y_coord[:, M - 1])
    elif bc_choice == 2:
        g_values = generate_grid_values(lambda x_val, y_val: g_neumann(x_val, y_val, R), x_coord[:, M - 1], y_coord[:, M - 1])

    if bc_choice == 2:
        u_fourier_0_arr = compute_zero_mode(u_true, iAngle, method_cfg["azu_unif"])
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    u_approx = poisson_solver(
        f_values, g_values, u_fourier_0,
        N, M, iRadius, iAngle, R,
        quad_rule, bc_choice,
        rad_unif, azu_unif,
        use_nudft_angular=use_nudft,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps_finufft=eps_finufft,
        num_processors=num_processors,
        use_gpu=actual_use_gpu,
        **kwargs,
    )
    return x_coord, y_coord, u_approx, u_true

def run_table_1(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R,
                num_processors=None, use_gpu=None, time_trials=None, num_runs=None,
                maxiter_nufft=50, tol_nufft=1e-8, reg_param=1e-12, eps_finufft=1e-12, **kwargs):
    # Dummy Warmup Solve (Warms up thread pools, CPU cache, and GPU plans)
    try:
        if methods and N_values and M_values:
            run_single_case(N=N_values[0], M=M_values[0], method_cfg=methods[0], bc_name="dirichlet", quad_name="trapezoidal",
                            u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                            num_processors=num_processors, use_gpu=use_gpu, time_trials=False, num_runs=1,
                            maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft, reg_param=reg_param, eps_finufft=eps_finufft, **kwargs)
    except Exception:
        pass

    table1_results = []
    for method in methods:
        for N in N_values:
            for M in M_values:
                res = run_single_case(
                    N=N, M=M, method_cfg=method, bc_name="dirichlet", quad_name="trapezoidal",
                    u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                    num_processors=num_processors, use_gpu=use_gpu, time_trials=time_trials, num_runs=num_runs,
                    maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft, reg_param=reg_param, eps_finufft=eps_finufft, **kwargs
                )
                table1_results.append(res)
    return pd.DataFrame(table1_results)

def run_table_2(methods, N_fixed, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R,
                num_processors=None, use_gpu=None, time_trials=None, num_runs=None,
                maxiter_nufft=50, tol_nufft=1e-8, reg_param=1e-12, eps_finufft=1e-12, **kwargs):
    # Dummy Warmup Solve
    try:
        if methods and M_values:
            run_single_case(N=N_fixed, M=M_values[0], method_cfg=methods[0], bc_name="dirichlet", quad_name="trapezoidal",
                            u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                            num_processors=num_processors, use_gpu=use_gpu, time_trials=False, num_runs=1,
                            maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft, reg_param=reg_param, eps_finufft=eps_finufft, **kwargs)
    except Exception:
        pass

    table2_results = []
    for method in methods:
        for M in M_values:
            for quad_name in ["trapezoidal", "simpson"]:
                for bc_name in ["dirichlet", "neumann"]:
                    res = run_single_case(
                        N=N_fixed, M=M, method_cfg=method, bc_name=bc_name, quad_name=quad_name,
                        u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                        num_processors=num_processors, use_gpu=use_gpu, time_trials=time_trials, num_runs=num_runs,
                        maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft, reg_param=reg_param, eps_finufft=eps_finufft, **kwargs
                    )
                    table2_results.append(res)
    return pd.DataFrame(table2_results)

def display_table_1(df_table1, methods, N_values, M_values):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"

    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : TABLE 1\n{'='*80}")
        display(df_table1[df_table1["method"] == name].pivot(index="N", columns="M", values="L_inf_rel").reindex(index=N_values, columns=M_values).map(dash_if_nan))

def display_timing(df_table1, methods, N_values, M_values):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"

    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : TABLE 1 (Timing)\n{'='*80}")
        display(df_table1[df_table1["method"] == name].pivot(index="N", columns="M", values="time").reindex(index=N_values, columns=M_values).map(dash_if_nan))

def display_timing_2(df_table2, methods, N_values, M_values):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"

    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : TABLE 2 (Timing)\n{'='*80}")
        df2 = df_table2[df_table2["method"] == name]
        display(pd.concat({(q.capitalize() + " rule", b.capitalize(), "Time"): df2[(df2["quad"] == q) & (df2["bc"] == b)].set_index("M")["time"] for q in ["trapezoidal", "simpson"] for b in ["dirichlet", "neumann"]}, axis=1).reindex(M_values).map(dash_if_nan))

def display_table_2(df_table2, methods, N_values, M_values):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"

    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : TABLE 2\n{'='*80}")
        df2 = df_table2[df_table2["method"] == name]
        display(pd.concat({(q.capitalize() + " rule", b.capitalize(), m): df2[(df2["quad"] == q) & (df2["bc"] == b)].set_index("M")["m"] for q in ["trapezoidal", "simpson"] for b in ["dirichlet", "neumann"] for m in ["L_inf_rel", "L2_rel"]}, axis=1).reindex(M_values).map(dash_if_nan))


def display_table_varying_M(df_table, methods, M_values, title="TABLE"):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"
    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : {title}\n{'='*80}")
        df2 = df_table[df_table["method"] == name]
        display(pd.concat({(q.capitalize() + " rule", b.capitalize(), m): df2[(df2["quad"] == q) & (df2["bc"] == b)].set_index("M")["m"] for q in ["trapezoidal", "simpson"] for b in ["dirichlet", "neumann"] for m in ["L_inf_rel", "L2_rel"]}, axis=1).reindex(M_values).map(dash_if_nan))

def setup_problem_5(alpha=5):
    x, y = sp.symbols('x y')
    u_sym = sp.sin(alpha * sp.pi * (x + y))
    return get_problem_functions(u_sym, x, y)

def setup_problem_6():
    x, y = sp.symbols('x y')
    phi_x = sp.exp(-100 * (x - 0.5)**2) * (x**2 - x)
    phi_y = sp.exp(-100 * (y - 0.5)**2) * (y**2 - y)
    u_sym = 10 * phi_x * phi_y
    return get_problem_functions(u_sym, x, y)

def setup_problem_7():
    # Generic setup pattern for Problem 7
    x, y = sp.symbols('x y')
    u_sym = sp.cos(10 * sp.pi * x) * sp.cos(10 * sp.pi * y)
    return get_problem_functions(u_sym, x, y)

def run_timing_analysis(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R,
                        num_processors=None, use_gpu=None, time_trials=None, num_runs=None,
                        maxiter_nufft=50, tol_nufft=1e-8, reg_param=1e-12, eps_finufft=1e-12, **kwargs):
    timing_results = []
    for method in methods:
        for N in N_values:
            for M in M_values:
                res = run_single_case(
                    N=N, M=M, method_cfg=method, bc_name="dirichlet", quad_name="trapezoidal",
                    u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                    num_processors=num_processors, use_gpu=use_gpu, time_trials=time_trials, num_runs=num_runs,
                    maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft, reg_param=reg_param, eps_finufft=eps_finufft, **kwargs
                )
                timing_results.append({
                    "method": method["name"],
                    "label": method["label"],
                    "N": N,
                    "M": M,
                    "time": res.get("time", np.nan)
                })
    return pd.DataFrame(timing_results)

def display_timing_results(df_timing, methods, N_values, M_values):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.4f} s"

    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : TIMING VS (N, M)\n{'='*80}")
        df2 = df_timing[df_timing["method"] == name]
        display(df2.pivot(index="N", columns="M", values="time").reindex(index=N_values, columns=M_values).map(dash_if_nan))
        display(pd.concat({(q.capitalize() + " rule", b.capitalize(), m): df2[(df2["quad"] == q) & (df2["bc"] == b)].set_index("M")[m] for q in ["trapezoidal", "simpson"] for b in ["dirichlet", "neumann"] for m in ["L_inf_rel", "L2_rel", "time"]}, axis=1).reindex(M_values).map(dash_if_nan))


from tqdm.auto import tqdm

def run_table_1_tracked(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R,
                        num_processors=None, use_gpu=None, time_trials=None, num_runs=None,
                        maxiter_nufft=50, tol_nufft=1e-8, reg_param=1e-12, eps_finufft=1e-12,
                        desc="Running", position=1, **kwargs):
    table1_results = []
    total = len(methods) * len(N_values) * len(M_values)

    pbar = tqdm(total=total, desc=desc, unit="case", position=position, leave=False)
    for method in methods:
        for N in N_values:
            for M in M_values:
                case_start = time.perf_counter()
                res = run_single_case(
                    N=N, M=M, method_cfg=method, bc_name="dirichlet", quad_name="trapezoidal",
                    u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann,
                    BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                    num_processors=num_processors, use_gpu=use_gpu, time_trials=time_trials, num_runs=num_runs,
                    maxiter_nufft=maxiter_nufft, tol_nufft=tol_nufft, reg_param=reg_param, eps_finufft=eps_finufft, **kwargs
                )
                wall_time = time.perf_counter() - case_start
                res["wall_time"] = wall_time
                table1_results.append(res)

                pbar.set_postfix({"method": method["name"], "N": N, "M": M, "t": f"{wall_time:.2f}s"})
                pbar.update(1)
    pbar.close()

    return pd.DataFrame(table1_results)