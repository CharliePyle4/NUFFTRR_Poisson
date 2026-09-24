from tqdm.asyncio import tqdm
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
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
from Poisson_Solver.visualization import (
    compute_error_metrics,
    plot_on_disk,
    plot_on_disk_with_error,
)
from Poisson_Solver.poisson_solver import poisson_solver

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

def setup_problem_7(use_paper_smooth_formula=False):
    """
    Problem 7 (Borges & Daripa JCP):
    Discontinuous boundary conditions on the unit disk B(0; 1):
      Delta u = f in B(0; 1)
      u = g on d B
    where:
      f(x, y) = -4*(x**2*y + y**3)*sin(1 - x**2 - y**2) - 8*y*cos(1 - x**2 - y**2)
      g(e^{i*alpha}) = 0 for alpha in (0, pi), 1 for alpha in (pi, 2*pi), 1/2 for alpha in {0, pi, 2*pi}
    Exact solution:
      u(x, y) = 1/2 + y*sin(1 - x**2 - y**2) - (2/pi)*harmonic
      where harmonic = sum_{k=1}^infty r^{2k-1}*sin((2k-1)*alpha)/(2k-1)
                     = 0.5 * atan2(2*y, 1 - x**2 - y**2)
    """
    def u(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r2 = x**2 + y**2
        r = np.sqrt(r2)
        alpha = np.arctan2(y, x) % (2.0 * np.pi)

        if use_paper_smooth_formula:
            smooth_part = np.sin(y * (1.0 - r2))
        else:
            smooth_part = y * np.sin(1.0 - r2)

        with np.errstate(divide='ignore', invalid='ignore'):
            harmonic = 0.5 * np.arctan2(2.0 * y, 1.0 - r2)
            at_b = np.isclose(r, 1.0, atol=1e-12)
            b_val = np.where(
                np.isclose(alpha, 0.0) | np.isclose(alpha, np.pi) | np.isclose(alpha, 2.0 * np.pi),
                0.0,
                np.where((alpha > 0.0) & (alpha < np.pi), np.pi / 4.0, -np.pi / 4.0)
            )
            harmonic = np.where(at_b, b_val, harmonic)

        return 0.5 + smooth_part - (2.0 / np.pi) * harmonic

    def f(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r2 = x**2 + y**2
        return -4.0 * (x**2 * y + y**3) * np.sin(1.0 - r2) - 8.0 * y * np.cos(1.0 - r2)

    def g_dirichlet(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        alpha = np.arctan2(y, x) % (2.0 * np.pi)
        return np.where(
            np.isclose(alpha, 0.0) | np.isclose(alpha, np.pi) | np.isclose(alpha, 2.0 * np.pi),
            0.5,
            np.where((alpha > 0.0) & (alpha < np.pi), 0.0, 1.0)
        )

    def g_neumann(x, y, R):
        return np.zeros_like(x)

    return u, f, g_dirichlet, g_neumann

problem_7_setup = setup_problem_7

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

def run_single_case(N, M, method_cfg, bc_name, quad_name, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, num_processors=None, use_gpu=False, **kwargs):
    bc_choice = BC_MAP[bc_name]
    quad_rule = QUAD_MAP[quad_name]

    azu_unif = method_cfg["azu_unif"]
    use_nudft = method_cfg["use_nudft"]
    grid_type = method_cfg.get("grid_type", 1 if azu_unif == 2 else azu_unif)

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
        u_fourier_0_arr = compute_zero_mode(u_true, iAngle, method_cfg["azu_unif"], num_processors=num_processors)
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    try:
        start_time = time.perf_counter()
        u_approx = poisson_solver(
            f_values, g_values, u_fourier_0,
            N, M, iRadius, iAngle, R,
            quad_rule, bc_choice,
            rad_unif, grid_type,
            use_nudft_angular=(use_nudft if use_nudft is not None else False),
            maxiter_nufft=50,
            tol_nufft=1e-8,
            num_processors=num_processors,
            use_gpu=use_gpu,
            **kwargs
        )
        solve_time = time.perf_counter() - start_time

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

def solve_for_grids(N, M, method_cfg, bc_name, quad_name, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, num_processors=None, use_gpu=False, **kwargs):
    bc_choice = BC_MAP[bc_name]
    quad_rule = QUAD_MAP[quad_name]

    azu_unif = method_cfg["azu_unif"]
    use_nudft = method_cfg.get("use_nudft", False)
    grid_type = method_cfg.get("grid_type", 1 if azu_unif == 2 else azu_unif)

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
        u_fourier_0_arr = compute_zero_mode(u_true, iAngle, method_cfg["azu_unif"], num_processors=num_processors)
        u_fourier_0 = u_fourier_0_arr[-1]
    else:
        u_fourier_0 = np.array([])

    u_approx = poisson_solver(
        f_values, g_values, u_fourier_0,
        N, M, iRadius, iAngle, R,
        quad_rule, bc_choice,
        rad_unif, grid_type,
        use_nudft_angular=use_nudft,
        maxiter_nufft=50,
        tol_nufft=1e-8,
        num_processors=num_processors,
        use_gpu=use_gpu,
        **kwargs
    )
    return x_coord, y_coord, u_approx, u_true

def run_table_1(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, **kwargs):
    table1_results = []
    for method in methods:
        for N in N_values:
            for M in M_values:
                res = run_single_case(
                    N=N, M=M, method_cfg=method, bc_name="dirichlet", quad_name="trapezoidal",
                    u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                    **kwargs
                )
                table1_results.append(res)
    return pd.DataFrame(table1_results)


def run_table_10(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, mask_radius=0.01, **kwargs):
    """
    Run Table X for Problem 7:
    Relative errors in norm ||·||_∞ evaluated over B(0; 1) - (B_{mask_radius}(1, 0) U B_{mask_radius}(-1, 0)).
    """
    table10_results = []
    for method in methods:
        for N in N_values:
            for M in M_values:
                start_time = time.perf_counter()
                x_coord, y_coord, u_approx, u_true = solve_for_grids(
                    N=N, M=M, method_cfg=method, bc_name="dirichlet", quad_name="trapezoidal",
                    u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann,
                    BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                    **kwargs
                )
                solve_time = time.perf_counter() - start_time
                err = np.abs(u_approx - u_true)

                # Standard error metrics
                iRadius = build_radial_mesh(M, rad_unif, R)
                iAngle = get_cached_angle_mesh(method, N, M)
                _, linf_rel, _, l2_rel = compute_error_metrics(u_approx, u_true, iRadius, iAngle)

                # Masked error excluding points within mask_radius of (1, 0) and (-1, 0)
                dist_p1 = np.sqrt((x_coord - 1.0)**2 + y_coord**2)
                dist_m1 = np.sqrt((x_coord + 1.0)**2 + y_coord**2)
                valid = (dist_p1 >= mask_radius) & (dist_m1 >= mask_radius)
                u_max_valid = np.max(np.abs(u_true[valid]))
                linf_rel_masked = np.max(err[valid]) / u_max_valid if u_max_valid > 0 else np.max(err[valid])

                table10_results.append({
                    "method": method["name"],
                    "label": method["label"],
                    "N": N,
                    "M": M,
                    "bc": "dirichlet",
                    "quad": "trapezoidal",
                    "L_inf_rel": linf_rel,
                    "L_inf_rel_masked": linf_rel_masked,
                    "L2_rel": l2_rel,
                    "time": solve_time,
                })
    return pd.DataFrame(table10_results)


def display_table_10(df_table10, methods, N_values, M_values, value_col="L_inf_rel_masked"):
    """
    Display Table X for Problem 7: Relative Errors in Norm ||·||_∞.
    """
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"

    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : TABLE X (Problem 7 - Relative Errors in Norm ||.||_inf)\n{'='*80}")
        display(df_table10[df_table10["method"] == name].pivot(index="N", columns="M", values=value_col).reindex(index=N_values, columns=M_values).map(dash_if_nan))


def plot_problem_7_1d_section(solutions_dict, R=1.0, figsize=(14, 5)):
    """
    Plot Figure 15: Errors on the 1D section from (0, -1) to (0, 1).
    (a) Linear plot showing convergence as N increases (64, 128, 256).
    (b) Log-scaling plot showing convergence rate.

    solutions_dict: dict mapping N -> (x_coord, y_coord, u_approx, u_true)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for N in sorted(solutions_dict.keys()):
        x_c, y_c, u_app, u_tr = solutions_dict[N]
        err = np.abs(u_tr - u_app.real if np.iscomplexobj(u_app) else u_tr - u_app)
        M = x_c.shape[1]
        r_m = np.linspace(0, R, M)

        # Segment from (0, -1) to (0, 1):
        # theta = 3*pi/2 for s in [-1, 0] (j = 3*N//4)
        # theta = pi/2 for s in [0, 1] (j = N//4)
        s_coords = np.concatenate([-r_m[::-1], r_m[1:]])
        err_slice = np.concatenate([err[3 * N // 4, :][::-1], err[N // 4, 1:]])

        ax1.plot(s_coords, err_slice, label=f"N = {N}", linewidth=1.5)
        ax2.plot(s_coords, np.maximum(err_slice, 1e-16), label=f"N = {N}", linewidth=1.5)

    ax1.set_xlabel("Radial position", fontsize=11)
    ax1.set_ylabel("Error", fontsize=11)
    ax1.set_title("(a) Convergence as Fourier coefficients increase (Linear)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.set_xlabel("Radial position", fontsize=11)
    ax2.set_ylabel("Error (log scale)", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_title("(b) Errors observed in log-scaling", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle("Figure 15: Problem 7—Errors along the one-dimensional section from (0, -1) to (0, 1)", fontsize=13)
    plt.tight_layout()
    plt.show()

def run_table_2(methods, N_fixed, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, **kwargs):
    table2_results = []
    for method in methods:
        for M in M_values:
            for quad_name in ["trapezoidal", "simpson"]:
                for bc_name in ["dirichlet", "neumann"]:
                    res = run_single_case(
                        N=N_fixed, M=M, method_cfg=method, bc_name=bc_name, quad_name=quad_name,
                        u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                        **kwargs
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
        display(pd.concat({(q.capitalize() + " rule", b.capitalize(), m): df2[(df2["quad"] == q) & (df2["bc"] == b)].set_index("M")[m] for q in ["trapezoidal", "simpson"] for b in ["dirichlet", "neumann"] for m in ["L_inf_rel", "L2_rel"]}, axis=1).reindex(M_values).map(dash_if_nan))


def display_table_varying_M(df_table, methods, M_values, title="TABLE"):
    def dash_if_nan(x):
        return "—" if pd.isna(x) else f"{x:.1e}"
    for method in methods:
        name = method["name"]
        print(f"\n{'='*80}\n{method['label']} : {title}\n{'='*80}")
        df2 = df_table[df_table["method"] == name]
        display(pd.concat({(q.capitalize() + " rule", b.capitalize(), m): df2[(df2["quad"] == q) & (df2["bc"] == b)].set_index("M")[m] for q in ["trapezoidal", "simpson"] for b in ["dirichlet", "neumann"] for m in ["L_inf_rel", "L2_rel"]}, axis=1).reindex(M_values).map(dash_if_nan))

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

def run_timing_analysis(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, **kwargs):
    timing_results = []
    for method in methods:
        for N in N_values:
            for M in M_values:
                res = run_single_case(
                    N=N, M=M, method_cfg=method, bc_name="dirichlet", quad_name="trapezoidal",
                    u=u, f=f, g_dirichlet=g_dirichlet, g_neumann=g_neumann, BC_MAP=BC_MAP, QUAD_MAP=QUAD_MAP, rad_unif=rad_unif, R=R,
                    **kwargs
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

def run_table_1_tracked(methods, N_values, M_values, u, f, g_dirichlet, g_neumann, BC_MAP, QUAD_MAP, rad_unif, R, desc="Running", position=1, **kwargs):
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
                    **kwargs
                )
                wall_time = time.perf_counter() - case_start
                res["wall_time"] = wall_time
                table1_results.append(res)

                pbar.set_postfix({"method": method["name"], "N": N, "M": M, "t": f"{wall_time:.2f}s"})
                pbar.update(1)
    pbar.close()

    return pd.DataFrame(table1_results)