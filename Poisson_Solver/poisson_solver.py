import numpy as np
import finufft
from pynufft import NUFFT


def poisson_solver(f_values, g_values, u_fourier_0,
                   N, M, r_m, theta_j, R,
                   quad_rule, BC_choice,
                   rad_unif, grid_type,
                   use_nudft_angular: bool = False,
                   maxiter_nufft: int = 50,
                   tol_nufft: float = 1e-8,
                   reg_param: float = 1e-12,
                   eps_finufft: float = 1e-12,
                   precond_shift: float = 1e-3,
                   kde_oversample: int = 4,
                   kde_bandwidth: float = 1.0,
                   num_processors: int = None,
                   use_gpu: bool = False,
                   **kwargs):
    """
    Solve Δu = f on a disk of radius R in polar coords using Fourier-in-θ
    and radial integration (C, D).

    grid_type:
        1 -> Uniform angular grid in θ (standard FFT).
        2, 3 -> Shared non-uniform angular grid in θ (NUFFT / NUDFT).

    use_nudft_angular:
        Only used when grid_type in (2, 3) (nonuniform angles).
        False (default) -> NUFFT + block CG (fast).
        True            -> direct NUDFT solve (dense, reference).

    num_processors:
        Number of threads/processors to use for CPU parallel FFTW / FINUFFT execution.
        Defaults to None (all available CPU cores).
    """

    if use_gpu:
        from .gpu_solver.poisson_solver import poisson_solver as backend_solver
    else:
        from .cpu_solver.poisson_solver import poisson_solver as backend_solver

    return backend_solver(
        f_values=f_values,
        g_values=g_values,
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
        precond_shift=precond_shift,
        kde_oversample=kde_oversample,
        kde_bandwidth=kde_bandwidth,
        num_processors=num_processors,
        **kwargs
    )
