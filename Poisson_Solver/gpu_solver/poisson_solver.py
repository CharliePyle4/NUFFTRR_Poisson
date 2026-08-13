import cupy as cp

from .fourier.fourier import (
    compute_angular_fourier_coefficients,
    synthesize_spatial_from_fourier,
    compute_u_fourier_coefficients,
)

from .radial.radial import (
    compute_v_neg_pos,
    combine_v_neg_pos_to_v,
    compute_radial_integrals
)


def poisson_solver(f_values, g_values, u_fourier_0,
                   N, M, r_m, theta_j, R,
                   quad_rule, BC_choice,
                   rad_unif, grid_type: int = 1,
                   use_nudft_angular: bool = False,
                   maxiter_nufft: int = 50,
                   tol_nufft: float = 1e-8,
                   reg_param: float = 1e-12,
                   eps_finufft: float = 1e-12,
                   precond_shift: float = 1e-3,
                   kde_oversample: int = 4,
                   kde_bandwidth: float = 1.0,
                   **kwargs):
    """
    GPU-accelerated solver for Δu = f on a disk of radius R using CuPy / cuFFT / cuFINUFFT.

    grid_type:
        1 -> Uniform angular grid in θ (standard FFT).
        2, 3 -> Shared non-uniform angular grid in θ (NUFFT / NUDFT).

    use_nudft_angular:
        Only used when grid_type in (2, 3) (nonuniform angles).
        False (default) -> cuFINUFFT + block CG / PCGLS (fast).
        True            -> direct GPU NUDFT solve.
    """
    # Map legacy azu_unif alias if passed
    if "azu_unif" in kwargs:
        azu = kwargs.pop("azu_unif")
        if azu == 2:
            grid_type = 1
        elif azu in (1, 3):
            grid_type = 3

    # Ensure arrays are on GPU device
    f_gpu = cp.asarray(f_values)
    g_gpu = cp.asarray(g_values)
    r_gpu = cp.asarray(r_m)
    th_gpu = cp.asarray(theta_j) if theta_j is not None else None
    u0_gpu = cp.asarray(u_fourier_0) if u_fourier_0 is not None else complex(0.0)

    # Step 1: angular Fourier coefficients
    f_fourier_coeff, g_fourier_coeff = compute_angular_fourier_coefficients(
        f_values=f_gpu,
        g_values=g_gpu,
        theta_j=th_gpu,
        grid_type=grid_type,
        use_nudft_angular=use_nudft_angular,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
        reg_param=reg_param,
        eps=eps_finufft,
        precond_shift=precond_shift,
        kde_oversample=kde_oversample,
        kde_bandwidth=kde_bandwidth,
        **kwargs
    )

    # Step 2: radial integrals C_n and D_n
    C, D = compute_radial_integrals(
        r_m=r_gpu,
        f_fourier_coeff=f_fourier_coeff,
        quad_rule=quad_rule,
        rad_unif=rad_unif,
    )

    # Steps 3–4
    v_neg, v_pos = compute_v_neg_pos(C, D, r_gpu, N, M, quad_rule)

    # Step 5
    v = combine_v_neg_pos_to_v(v_neg, v_pos, r_gpu, N, M)

    # Step 6
    u_fourier_coeff = compute_u_fourier_coefficients(
        v=v,
        g_fourier_coeff=g_fourier_coeff,
        u_fourier_0=u0_gpu,
        N=N,
        M=M,
        r_m=r_gpu,
        R=R,
        BC_choice=BC_choice,
    )

    # Step 7: synthesis
    u_approx_gpu = synthesize_spatial_from_fourier(
        u_fourier_coeff=u_fourier_coeff,
        theta_j=th_gpu,
        N=N,
        grid_type=grid_type,
        eps=eps_finufft,
    )

    return u_approx_gpu
