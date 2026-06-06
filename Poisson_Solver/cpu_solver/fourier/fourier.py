import numpy as np
import finufft

from .uniform import compute_fourier_coeff_unif
from .nonuniform import (
    compute_fourier_coeff_nonunif,
    compute_fourier_coeff_nonunif_perradius,
    _wrap_angles
)

#-----------------------------------------
# Analysis dispatcher
# ---------------------------------------------------------
def compute_angular_fourier_coefficients(f_values: np.ndarray,
                                         g_values: np.ndarray,
                                         theta_j,
                                         azu_unif: int,
                                         use_nudft_angular: bool = True,
                                         maxiter_nufft: int = 50,
                                         tol_nufft: float = 1e-8):
    """
    Compute angular Fourier coefficients (analysis step) for f and g.

    azu_unif:
      2 → uniform FFT
      1 → shared nonuniform mesh, theta_j shape (N,)
      0 → per-radius nonuniform mesh, theta_j shape (N, M)
    """
    f_values = np.asarray(f_values)
    g_values = np.asarray(g_values)

    if azu_unif == 2:
        # Uniform angles → standard FFT-based coefficients
        f_fc = compute_fourier_coeff_unif(f_values)
        g_fc = compute_fourier_coeff_unif(g_values)
        return f_fc, g_fc

    elif azu_unif == 1:
        # Nonuniform but shared mesh: theta_j is 1D of length N
        theta = np.asarray(theta_j, dtype=float)
        if theta.ndim != 1 or theta.size != f_values.shape[0]:
            raise ValueError(
                "For azu_unif == 1, theta_j must be 1D of length N "
                "matching the first dimension of f_values"
            )

        f_fc = compute_fourier_coeff_nonunif(
            f_values,
            theta,
            maxiter=maxiter_nufft,
            tol=tol_nufft,
            use_nudft=use_nudft_angular,
        )
        g_fc = compute_fourier_coeff_nonunif(
            g_values,
            theta,
            maxiter=maxiter_nufft,
            tol=tol_nufft,
            use_nudft=use_nudft_angular,
        )
        return f_fc, g_fc

    elif azu_unif == 0:
        # Fully nonuniform: theta_j is (N, M), different mesh at each radius.
        theta = np.asarray(theta_j, dtype=float)

        if f_values.ndim != 2:
            raise ValueError("For azu_unif == 0, f_values must have shape (N, M)")

        N, M = f_values.shape
        if theta.shape != (N, M):
            raise ValueError(
                f"For azu_unif == 0, theta_j must have shape (N, M) = ({N}, {M}), "
                f"got {theta.shape}"
            )

        # f: full grid (N, M) → per-radius NUDFT/NUFFT
        f_fc = compute_fourier_coeff_nonunif_perradius(
            f_values,
            theta,
            maxiter=maxiter_nufft,
            tol=tol_nufft,
            use_nudft=use_nudft_angular,
        )

        # g: boundary data only on r = R, typically shape (N,)
        # use the angular mesh at the outer radius (last column of theta)
        if g_values.ndim == 1:
            if g_values.shape[0] != N:
                raise ValueError(
                    "For azu_unif == 0 with 1D g_values, length must be N"
                )
            g_fc = compute_fourier_coeff_nonunif(
                g_values,
                theta[:, -1],
                maxiter=maxiter_nufft,
                tol=tol_nufft,
                use_nudft=use_nudft_angular,
            )
        else:
            # If you ever store g on all radii as (N, M), you can also do per-radius here.
            if g_values.shape != (N, M):
                raise ValueError(
                    "g_values must be either (N,) or (N,M) when azu_unif == 0"
                )
            g_fc = compute_fourier_coeff_nonunif_perradius(
                g_values,
                theta,
                maxiter=maxiter_nufft,
                tol=tol_nufft,
                use_nudft=use_nudft_angular,
            )

        return f_fc, g_fc

    else:
        raise ValueError('Incorrect index for "azu_unif"')


# ---------------------------------------------------------
# Synthesis dispatcher
# ---------------------------------------------------------
def synthesize_spatial_from_fourier(u_fourier_coeff: np.ndarray,
                                    theta_j,
                                    N: int,
                                    azu_unif: int,
                                    eps: float = 1e-12) -> np.ndarray:
    """
    azu_unif == 2: uniform IFFT
    azu_unif == 1: shared nonuniform, NUFFT-2, theta_j (N,)
    azu_unif == 0: per-radius nonuniform, NUFFT-2 loop, theta_j (N, M)
    """
    u_fourier_coeff = np.asarray(u_fourier_coeff)
    Np1, M = u_fourier_coeff.shape
    if Np1 != N + 1:
        raise ValueError("u_fourier_coeff must have shape (N+1, M)")

    halfN = N // 2

    if azu_unif == 2:
        coeff    = np.vstack([u_fourier_coeff[halfN:N, :],
                              u_fourier_coeff[0:halfN, :]])
        u_approx = np.fft.ifft(coeff, axis=0) * N
        return np.vstack([u_approx[1:, :], u_approx[:1, :]])

    elif azu_unif == 1:
        theta = np.asarray(theta_j, dtype=float)
        if theta.ndim != 1 or theta.size != N:
            raise ValueError("theta_j must be 1D of length N when azu_unif == 1")
        x        = np.ascontiguousarray(_wrap_angles(theta))
        coeff    = u_fourier_coeff[:N, :].copy()
        coeff[0, :] *= 2.0
        coeff_KN = np.ascontiguousarray(coeff.T, dtype=np.complex128)  # (M, N)
        out_KM   = finufft.nufft1d2(x, coeff_KN, isign=+1, eps=eps)   # (M, N)
        return out_KM.T                                                # (N, M)

    elif azu_unif == 0:
        theta = np.asarray(theta_j, dtype=float)
        if theta.shape != (N, M):
            raise ValueError("theta_j must have shape (N, M) when azu_unif == 0")
        base_coeff = u_fourier_coeff[:N, :].copy()
        base_coeff[0, :] *= 2.0
        u_approx = np.zeros((N, M), dtype=np.complex128)
        for ell in range(M):
            x      = np.ascontiguousarray(_wrap_angles(theta[:, ell]))
            fj_KN  = base_coeff[:, ell][None, :].astype(np.complex128)
            out_KM = finufft.nufft1d2(x, fj_KN, isign=+1, eps=eps)    # (1, N)
            u_approx[:, ell] = out_KM[0, :]
        return u_approx

    else:
        raise ValueError('Incorrect index for "azu_unif"')






def compute_u_fourier_coefficients(v: np.ndarray,
                                   g_fourier_coeff: np.ndarray,
                                   u_fourier_0: complex,
                                   N: int,
                                   M: int,
                                   r_m: np.ndarray,
                                   R: float,
                                   BC_choice: int) -> np.ndarray:
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
    u_fourier_coeff : ndarray, shape (N+1, M)
        Fourier coefficients u_n(r_m).
    """
    halfN = N // 2
    u_fourier_coeff = np.zeros((N + 1, M), dtype=complex)

    # central bin (k = 0)
    if BC_choice == 1:  # Dirichlet
        u_fourier_coeff[halfN, :] = (
            v[halfN, :] + (g_fourier_coeff[halfN] - v[halfN, M - 1])
        )
    elif BC_choice == 2:  # Neumann
        u_fourier_coeff[halfN, :] = v[halfN, :] + (u_fourier_0 - v[halfN, :])
    else:
        raise ValueError('Incorrect index for "BC_choice"')

    # all other modes
    n_idx = np.arange(N + 1)
    kabs_all = np.abs(n_idx - halfN)
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
