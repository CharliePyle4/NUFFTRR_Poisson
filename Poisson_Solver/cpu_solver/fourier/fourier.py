import scipy.special as sp_special
import finufft
import multiprocessing
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_fft
pyfftw.interfaces.cache.enable()
import numpy as np

from .uniform import compute_fourier_coeff_unif
from .nonuniform import (
    compute_fourier_coeff_nonunif,
    _wrap_angles
)

# ---------------------------------------------------------
# Analysis dispatcher
# ---------------------------------------------------------
def compute_angular_fourier_coefficients(f_values: np.ndarray,
                                         g_values: np.ndarray,
                                         theta_j,
                                         azu_unif: int,
                                         use_nudft_angular: bool = True,
                                         maxiter_nufft: int = 50,
                                         tol_nufft: float = 1e-8,
                                         reg_param: float = 1e-12,
                                         eps: float = 1e-12,
                                         precond_shift: float = 1e-3,
                                         kde_oversample: int = 4,
                                         kde_bandwidth: float = 1.0,
                                         **kwargs):
    """
    Compute angular Fourier coefficients (analysis step) for f and g.

    azu_unif:
      2 → uniform FFT
      1 → shared nonuniform mesh, theta_j shape (N,)
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

        # Unified Batching of [f, g] into a single NUFFT/NUDFT solve
        is_f_1d = (f_values.ndim == 1)
        is_g_1d = (g_values.ndim == 1)
        f_arr = f_values[:, None] if is_f_1d else f_values
        g_arr = g_values[:, None] if is_g_1d else g_values

        M_f = f_arr.shape[1]
        combined = np.ascontiguousarray(np.hstack([f_arr, g_arr]))

        combined_fc = compute_fourier_coeff_nonunif(
            combined,
            theta,
            maxiter=maxiter_nufft,
            tol=tol_nufft,
            use_nudft=use_nudft_angular,
            reg_param=reg_param,
            eps=eps,
            precond_shift=precond_shift,
            kde_oversample=kde_oversample,
            kde_bandwidth=kde_bandwidth,
        )

        f_fc = combined_fc[:, :M_f]
        g_fc = combined_fc[:, M_f:]

        return (f_fc[:, 0] if is_f_1d else f_fc), (g_fc[:, 0] if is_g_1d else g_fc)

    else:
        raise ValueError(
            f'Incorrect index for "azu_unif": {azu_unif}. '
            'Supported values are 2 (uniform) and 1 (shared nonuniform).'
        )


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
    """
    u_fourier_coeff = np.asarray(u_fourier_coeff)
    Np1, M = u_fourier_coeff.shape
    if Np1 != N + 1:
        raise ValueError("u_fourier_coeff must have shape (N+1, M)")

    halfN = N // 2

    if azu_unif == 2:
        n_threads = multiprocessing.cpu_count()
        
        coeff    = np.vstack([u_fourier_coeff[halfN:N, :],
                              u_fourier_coeff[0:halfN, :]])
        u_approx = fftw_fft.ifft(coeff, axis=0, threads=n_threads) * N
        return u_approx

    elif azu_unif == 1:
        theta = np.asarray(theta_j, dtype=float)
        if theta.ndim != 1 or theta.size != N:
            raise ValueError("theta_j must be 1D of length N when azu_unif == 1")
        x        = np.ascontiguousarray(_wrap_angles(theta))
        coeff    = u_fourier_coeff[:N, :].copy()
        coeff[0, :] += u_fourier_coeff[N, :]  # Recombine the split Nyquist mode (k = -N/2 and +N/2)
        coeff_KN = np.ascontiguousarray(coeff.T, dtype=np.complex128)  # (M, N)
        out_KM   = finufft.nufft1d2(x, coeff_KN, isign=+1, eps=eps)   # (M, N)
        return out_KM.T                                                # (N, M)

    else:
        raise ValueError(
            f'Incorrect index for "azu_unif": {azu_unif}. '
            'Supported values are 2 (uniform) and 1 (shared nonuniform).'
        )


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

    # Un-halve endpoint boundary coefficients so g_full[0] and g_full[N] match full v_n(R)
    g_full = g_fourier_coeff.copy()
    g_full[0] *= 2.0
    g_full[N] *= 2.0

    # central bin (k = 0)
    if BC_choice == 1:  # Dirichlet
        u_fourier_coeff[halfN, :] = (
            v[halfN, :] + (g_full[halfN] - v[halfN, M - 1])
        )
    elif BC_choice == 2:  # Neumann
        # v_0(R) is v[halfN, -1]. u_fourier_0 is the reference value u_0(R).
        # The difference is the constant C to add to the particular solution v_0(r).
        C = u_fourier_0 - v[halfN, -1]
        u_fourier_coeff[halfN, :] = v[halfN, :] + C
    else:
        raise ValueError('Incorrect index for "BC_choice"')

    # all other modes
    n_idx = np.arange(N + 1)
    kabs_all = np.abs(n_idx - halfN)
    mask = n_idx != halfN  # exclude central mode

    kabs = kabs_all[mask][:, None]       # (N, 1)
    ratio = (r_m / R)[None, :]          # (1, M)

    if BC_choice == 1:
        rp = ratio ** kabs                                    # compute once
        B = rp * (g_full[mask, None] - v[mask, M - 1][:, None])
        u_fourier_coeff[mask, :] = v[mask, :] + B
    elif BC_choice == 2:
        rp = ratio ** kabs                                    # compute once (was twice!)
        B = rp * ((R / kabs) * g_full[mask, None] + v[mask, M - 1][:, None])
        u_fourier_coeff[mask, :] = v[mask, :] + B

    return u_fourier_coeff
