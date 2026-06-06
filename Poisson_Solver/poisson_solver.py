import numpy as np
import finufft
from pynufft import NUFFT


def poisson_solver(f_values, g_values, u_fourier_0,
                   N, M, r_m, theta_j, R,
                   quad_rule, BC_choice,
                   rad_unif, azu_unif,
                   use_nudft_angular: bool = False,
                   maxiter_nufft: int = 50,
                   tol_nufft: float = 1e-8,
                   use_gpu: bool = False):
    """
    Solve Δu = f on a disk of radius R in polar coords using Fourier-in-θ
    and radial integration (C, D).

    use_nudft_angular:
        Only used when azu_unif == 0 (nonuniform angles).
        False (default) -> NUFFT + block CG (fast).
        True            -> direct NUDFT solve (dense, reference).
    """

    # Dynamically import the required solver backend functions
    if use_gpu:
        from .gpu_solver.fourier import (
            compute_angular_fourier_coefficients,
            synthesize_spatial_from_fourier,
            compute_u_fourier_coefficients,
        )
        from .gpu_solver.radial import (
            compute_v_neg_pos,
            combine_v_neg_pos_to_v,
            compute_radial_integrals
        )
    else:
        from .cpu_solver.fourier.fourier import (
            compute_angular_fourier_coefficients,
            synthesize_spatial_from_fourier,
            compute_u_fourier_coefficients,
        )
        from .cpu_solver.Radial.radial import (
            compute_v_neg_pos,
            combine_v_neg_pos_to_v,
            compute_radial_integrals
        )

    # Step 1: angular Fourier coefficients
    f_fourier_coeff, g_fourier_coeff = compute_angular_fourier_coefficients(
        f_values=f_values,
        g_values=g_values,
        theta_j=theta_j,
        azu_unif=azu_unif,
        use_nudft_angular=use_nudft_angular,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
    )



    # Step 2: radial integrals C_n and D_n
    C, D = compute_radial_integrals(
        r_m=r_m,
        f_fourier_coeff=f_fourier_coeff,
        quad_rule=quad_rule,
        rad_unif=rad_unif,
    )


    # Steps 3–4
    v_neg, v_pos = compute_v_neg_pos(C, D, r_m, N, M, quad_rule)

    # Step 5
    v = combine_v_neg_pos_to_v(v_neg, v_pos, r_m, N, M)

    # Step 6
    u_fourier_coeff = compute_u_fourier_coefficients(
        v=v,
        g_fourier_coeff=g_fourier_coeff,
        u_fourier_0=u_fourier_0,
        N=N,
        M=M,
        r_m=r_m,
        R=R,
        BC_choice=BC_choice,
    )

    # Step 7: synthesis
    u_approx = synthesize_spatial_from_fourier(
        u_fourier_coeff=u_fourier_coeff,
        theta_j=theta_j,
        N=N,
        azu_unif=azu_unif,
        eps=1e-12,
    )

    return u_approx
