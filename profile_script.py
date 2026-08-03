import time
import sys
import numpy as np

from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_fixed_nonuniform_azimuthal,
    generate_uniform_azimuthal,
    generate_cartesian_grid_on_disk,
    generate_grid_values
)
from Poisson_Solver.cpu_solver.fourier.fourier import (
    compute_angular_fourier_coefficients,
    synthesize_spatial_from_fourier,
    compute_u_fourier_coefficients
)
from Poisson_Solver.cpu_solver.radial.radial import (
    compute_radial_integrals,
    compute_v_neg_pos,
    combine_v_neg_pos_to_v
)

def u_true(x, y):
    return 3 * np.exp(x + y) * (x - x**2) * (y - y**2) + 5

def f_rhs(x, y):
    return 6 * np.exp(x + y) * x * y * (-3 + x + y + x * y)

def g_dirichlet(x, y):
    return u_true(x, y)

def profile_run(N, M, azu_unif, use_nudft):
    R = 1.0
    r_m = generate_uniform_radial(M, R)
    if azu_unif == 2:
        theta_j = generate_uniform_azimuthal(N)
    else:
        theta_j = generate_fixed_nonuniform_azimuthal(N, kind='jittered')
    
    x_coord, y_coord = generate_cartesian_grid_on_disk(theta_j, r_m)
    f_values = generate_grid_values(f_rhs, x_coord, y_coord)
    g_values = generate_grid_values(g_dirichlet, x_coord[:, M-1], y_coord[:, M-1])
    u_fourier_0 = np.array([])

    # Step 1: Fourier Analysis
    t0 = time.perf_counter()
    f_fc, g_fc = compute_angular_fourier_coefficients(f_values, g_values, theta_j, azu_unif, use_nudft_angular=use_nudft)
    t_step1 = time.perf_counter() - t0

    # Step 2: Radial Integrals
    t0 = time.perf_counter()
    C, D = compute_radial_integrals(r_m, f_fc, quad_rule=1, rad_unif=1)
    t_step2 = time.perf_counter() - t0

    # Step 3-5: Radial Recurrences
    t0 = time.perf_counter()
    v_neg, v_pos = compute_v_neg_pos(C, D, r_m, N, M, quad_rule=1)
    v = combine_v_neg_pos_to_v(v_neg, v_pos, r_m, N, M)
    t_step3_5 = time.perf_counter() - t0

    # Step 6: Boundary Matching
    t0 = time.perf_counter()
    u_fc = compute_u_fourier_coefficients(v, g_fc, u_fourier_0, N, M, r_m, R, BC_choice=1)
    t_step6 = time.perf_counter() - t0

    # Step 7: Spatial Synthesis
    t0 = time.perf_counter()
    u_approx = synthesize_spatial_from_fourier(u_fc, theta_j, N, azu_unif, eps=1e-12)
    t_step7 = time.perf_counter() - t0

    total = t_step1 + t_step2 + t_step3_5 + t_step6 + t_step7
    return {
        'N': N, 'M': M, 
        'Step1': t_step1,
        'Step2': t_step2,
        'Step3_5': t_step3_5,
        'Step6': t_step6,
        'Step7': t_step7,
        'Total': total
    }

print("=== UNIFORM (FFT) PROFILE ===", flush=True)
for N, M in [(64, 64), (256, 256), (512, 512)]:
    res = profile_run(N, M, azu_unif=2, use_nudft=False)
    p1 = res['Step1']*100/res['Total']
    p2 = res['Step2']*100/res['Total']
    p3 = res['Step3_5']*100/res['Total']
    p7 = res['Step7']*100/res['Total']
    print(f"N={N:3d}, M={M:3d} | Total: {res['Total']:.5f}s | Step1(Analysis): {p1:4.1f}% | Step2(RadInt): {p2:4.1f}% | Step3-5(Recurr): {p3:4.1f}% | Step7(Synth): {p7:4.1f}%", flush=True)

print("\n=== FIXED NONUNIFORM (NUDFT) PROFILE ===", flush=True)
for N, M in [(64, 64), (256, 256), (512, 512)]:
    res = profile_run(N, M, azu_unif=1, use_nudft=True)
    p1 = res['Step1']*100/res['Total']
    p2 = res['Step2']*100/res['Total']
    p3 = res['Step3_5']*100/res['Total']
    p7 = res['Step7']*100/res['Total']
    print(f"N={N:3d}, M={M:3d} | Total: {res['Total']:.5f}s | Step1(Analysis): {p1:4.1f}% | Step2(RadInt): {p2:4.1f}% | Step3-5(Recurr): {p3:4.1f}% | Step7(Synth): {p7:4.1f}%", flush=True)

print("\n=== FIXED NONUNIFORM (NUFFT) PROFILE ===", flush=True)
for N, M in [(64, 64), (256, 256), (512, 512)]:
    res = profile_run(N, M, azu_unif=1, use_nudft=False)
    p1 = res['Step1']*100/res['Total']
    p2 = res['Step2']*100/res['Total']
    p3 = res['Step3_5']*100/res['Total']
    p7 = res['Step7']*100/res['Total']
    print(f"N={N:3d}, M={M:3d} | Total: {res['Total']:.5f}s | Step1(Analysis): {p1:4.1f}% | Step2(RadInt): {p2:4.1f}% | Step3-5(Recurr): {p3:4.1f}% | Step7(Synth): {p7:4.1f}%", flush=True)
