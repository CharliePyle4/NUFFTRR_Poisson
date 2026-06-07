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
                   rad_unif, azu_unif,
                   use_nudft_angular: bool = False,
                   maxiter_nufft: int = 50,
                   tol_nufft: float = 1e-8):
    """
    GPU-accelerated solver for Δu = f on a disk of radius R.

    This function orchestrates the GPU-based Poisson solver pipeline. It handles
    the transfer of data from CPU (NumPy) to GPU (CuPy), calls the various
    computational kernels, and transfers the final result back to the CPU.

    use_nudft_angular:
        (Not yet implemented for GPU)
    """
    # --- Transfer data from CPU (NumPy) to GPU (CuPy) ---
    f_values_gpu = cp.asarray(f_values)
    g_values_gpu = cp.asarray(g_values)
    u_fourier_0_gpu = cp.asarray(u_fourier_0)
    r_m_gpu = cp.asarray(r_m)
    # theta_j might not be used if azu_unif=2, but convert anyway for consistency
    theta_j_gpu = cp.asarray(theta_j)


    # Step 1: angular Fourier coefficients
    f_fourier_coeff, g_fourier_coeff = compute_angular_fourier_coefficients(
        f_values=f_values_gpu,
        g_values=g_values_gpu,
        theta_j=theta_j_gpu,
        azu_unif=azu_unif,
        use_nudft_angular=use_nudft_angular,
        maxiter_nufft=maxiter_nufft,
        tol_nufft=tol_nufft,
    )



    # Step 2: radial integrals C_n and D_n
    C, D = compute_radial_integrals(
        r_m=r_m_gpu,
        f_fourier_coeff=f_fourier_coeff,
        quad_rule=quad_rule,
        rad_unif=rad_unif,
    )


    # Steps 3–4
    v_neg, v_pos = compute_v_neg_pos(C, D, r_m_gpu, N, M, quad_rule)

    # Step 5
    v = combine_v_neg_pos_to_v(v_neg, v_pos, r_m_gpu, N, M)

    # Step 6
    u_fourier_coeff = compute_u_fourier_coefficients(
        v=v,
        g_fourier_coeff=g_fourier_coeff,
        u_fourier_0=u_fourier_0_gpu,
        N=N,
        M=M,
        r_m=r_m_gpu,
        R=R,
        BC_choice=BC_choice,
    )

    # Step 7: synthesis
    u_approx_gpu = synthesize_spatial_from_fourier(
        u_fourier_coeff=u_fourier_coeff,
        theta_j=theta_j_gpu,
        N=N,
        azu_unif=azu_unif,
        eps=1e-12,
    )

    # --- Transfer result from GPU (CuPy) back to CPU (NumPy) ---
    return cp.asnumpy(u_approx_gpu)
