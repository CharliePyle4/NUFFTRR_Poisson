import cupy as cp

from .uniform import compute_fourier_coeff_unif

#-----------------------------------------
# Analysis dispatcher
# ---------------------------------------------------------
def compute_angular_fourier_coefficients(f_values: cp.ndarray,
                                         g_values: cp.ndarray,
                                         theta_j,
                                         azu_unif: int,
                                         use_nudft_angular: bool = False,
                                         maxiter_nufft: int = 50,
                                         tol_nufft: float = 1e-8):
    """
    Compute angular Fourier coefficients (analysis step) for f and g.

    azu_unif:
      2 → uniform FFT
      1 → shared nonuniform mesh, theta_j shape (N,)
      0 → per-radius nonuniform mesh, theta_j shape (N, M)
    """
    f_values = cp.asarray(f_values)
    g_values = cp.asarray(g_values)

    if azu_unif == 2:
        # Uniform angles → standard FFT-based coefficients
        f_fc = compute_fourier_coeff_unif(f_values)
        g_fc = compute_fourier_coeff_unif(g_values)
        return f_fc, g_fc
    elif azu_unif == 1:
        # Nonuniform cases are not yet implemented for the GPU solver.
        raise NotImplementedError(
            "GPU solver does not yet support shared nonuniform angular grids (azu_unif=1)."
        )
    elif azu_unif == 0:
        # Nonuniform cases are not yet implemented for the GPU solver.
        raise NotImplementedError(
            "GPU solver does not yet support per-radius nonuniform angular grids (azu_unif=0)."
        )
    else:
        raise ValueError('Incorrect index for "azu_unif"')


# ---------------------------------------------------------
# Synthesis dispatcher
# ---------------------------------------------------------
def synthesize_spatial_from_fourier(u_fourier_coeff: cp.ndarray,
                                    theta_j,
                                    N: int,
                                    azu_unif: int,
                                    eps: float = 1e-12) -> cp.ndarray:
    """
    azu_unif == 2: uniform IFFT
    azu_unif == 1: shared nonuniform, NUFFT-2, theta_j (N,)
    azu_unif == 0: per-radius nonuniform, NUFFT-2 loop, theta_j (N, M)
    """
    u_fourier_coeff = cp.asarray(u_fourier_coeff)
    Np1, M = u_fourier_coeff.shape
    if Np1 != N + 1:
        raise ValueError("u_fourier_coeff must have shape (N+1, M)")

    halfN = N // 2

    if azu_unif == 2:
        coeff    = cp.vstack([u_fourier_coeff[halfN:N, :],
                              u_fourier_coeff[0:halfN, :]])
        u_approx = cp.fft.ifft(coeff, axis=0) * N
        return cp.vstack([u_approx[1:, :], u_approx[:1, :]])
    elif azu_unif == 1:
        # Nonuniform cases are not yet implemented for the GPU solver.
        raise NotImplementedError(
            "GPU solver does not yet support shared nonuniform angular grids (azu_unif=1)."
        )
    elif azu_unif == 0:
        # Nonuniform cases are not yet implemented for the GPU solver.
        raise NotImplementedError(
            "GPU solver does not yet support per-radius nonuniform angular grids (azu_unif=0)."
        )
    else:
        raise ValueError('Incorrect index for "azu_unif"')




def compute_u_fourier_coefficients(v: cp.ndarray,
                                   g_fourier_coeff: cp.ndarray,
                                   u_fourier_0: complex,
                                   N: int,
                                   M: int,
                                   r_m: cp.ndarray,
                                   R: float,
                                   BC_choice: int) -> cp.ndarray:
    """
    Compute u_n(r) Fourier coefficients from v_n(r) and boundary data.

    Parameters
    ----------
    v : ndarray, shape (N+1, M)
        Intermediate radial quantities for each Fourier mode.
    g_fourier_coeff : ndarray, shape (N+1,)
        Fourier coefficients of boundary data g (Dirichlet or Neumann data).
    u_fourier_0 : complex
        Central-mode constant for Neumann problem (ignored for Dirichlet).
    N : int
        Number of angular points.
    M : int
        Number of radial points.
    r_m : ndarray, shape (M,)
        Radial grid, with r_m[-1] = R.
    R : float
        Disk radius.
    BC_choice : int
        1 → Dirichlet, 2 → Neumann.

    Returns
    -------
    u_fourier_coeff : cp.ndarray, shape (N+1, M)
        Fourier coefficients u_n(r_m).
    """
    halfN = N // 2
    u_fourier_coeff = cp.zeros((N + 1, M), dtype=complex)

    # central bin (k = 0)
    if BC_choice == 1:  # Dirichlet
        u_fourier_coeff[halfN, :] = (
            v[halfN, :] + (g_fourier_coeff[halfN] - v[halfN, M - 1])
        )
    elif BC_choice == 2:  # Neumann
        # v_0(R) is v[halfN, -1]. u_fourier_0 is the reference value u_0(R).
        # The difference is the constant C to add to the particular solution v_0(r).
        C = u_fourier_0 - v[halfN, -1]
        u_fourier_coeff[halfN, :] = v[halfN, :] + C
    else:
        raise ValueError('Incorrect index for "BC_choice"')

    # all other modes
    n_idx = cp.arange(N + 1)
    kabs_all = cp.abs(n_idx - halfN)
    mask = n_idx != halfN  # exclude central mode

    kabs = kabs_all[mask][:, None]       # (N, 1)
    ratio = (r_m / R)[None, :]          # (1, M)

    if BC_choice == 1:
        B = ratio ** kabs * (
            g_fourier_coeff[mask, None] - v[mask, M - 1][:, None]
        )
        u_fourier_coeff[mask, :] = v[mask, :] + B
    elif BC_choice == 2:
        B = ratio ** kabs * (R / kabs) * g_fourier_coeff[mask, None] \
            + ratio ** kabs * v[mask, M - 1][:, None]
        u_fourier_coeff[mask, :] = v[mask, :] + B

    return u_fourier_coeff
