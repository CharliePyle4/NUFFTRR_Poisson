import numpy as np
import cupy as cp

try:
    import cufinufft
except ImportError:
    try:
        from finufft import cufinufft
    except ImportError:
        cufinufft = None

from .uniform import compute_fourier_coeff_unif
from .nonuniform import (
    compute_fourier_coeff_nonunif,
    _wrap_angles
)

# ---------------------------------------------------------
# Analysis dispatcher
# ---------------------------------------------------------
def compute_angular_fourier_coefficients(f_values: cp.ndarray,
                                         g_values: cp.ndarray,
                                         theta_j,
                                         grid_type: int = 1,
                                         use_nudft_angular: bool = False,
                                         maxiter_nufft: int = 50,
                                         tol_nufft: float = 1e-8,
                                         reg_param: float = 1e-12,
                                         eps: float = 1e-12,
                                         precond_shift: float = 1e-3,
                                         kde_oversample: int = 4,
                                         kde_bandwidth: float = 1.0,
                                         **kwargs):
    """
    Compute angular Fourier coefficients (analysis step) for f and g on GPU.

    grid_type:
      1 → uniform FFT
      2, 3 → shared nonuniform mesh, theta_j shape (N,)
    """
    # Map legacy azu_unif alias if passed
    if "azu_unif" in kwargs:
        azu = kwargs.pop("azu_unif")
        if azu == 2:
            grid_type = 1
        elif azu in (1, 3):
            grid_type = 3

    f_values = cp.asarray(f_values)
    g_values = cp.asarray(g_values)

    if grid_type == 1:
        # Uniform angles → standard FFT-based coefficients
        f_fc = compute_fourier_coeff_unif(f_values)
        g_fc = compute_fourier_coeff_unif(g_values)
        return f_fc, g_fc

    elif grid_type in (2, 3):
        # Nonuniform but shared mesh: theta_j is 1D of length N
        theta = cp.asarray(theta_j, dtype=float)
        if theta.ndim != 1 or theta.size != f_values.shape[0]:
            raise ValueError(
                "For grid_type 2 or 3, theta_j must be 1D of length N "
                "matching the first dimension of f_values"
            )

        # Unified Batching of [f, g] into a single cuFINUFFT/NUDFT solve
        is_f_1d = (f_values.ndim == 1)
        is_g_1d = (g_values.ndim == 1)
        f_arr = f_values[:, None] if is_f_1d else f_values
        g_arr = g_values[:, None] if is_g_1d else g_values

        M_f = f_arr.shape[1]
        combined = cp.ascontiguousarray(cp.hstack([f_arr, g_arr]))

        combined_fc = compute_fourier_coeff_nonunif(
            combined,
            theta,
            grid_type=grid_type,
            maxiter=maxiter_nufft,
            tol=tol_nufft,
            use_nudft=use_nudft_angular,
            reg_param=reg_param,
            eps=eps,
            precond_shift=precond_shift,
            kde_oversample=kde_oversample,
            kde_bandwidth=kde_bandwidth,
            **kwargs,
        )

        f_fc = combined_fc[:, :M_f]
        g_fc = combined_fc[:, M_f:]

        return (f_fc[:, 0] if is_f_1d else f_fc), (g_fc[:, 0] if is_g_1d else g_fc)

    else:
        raise ValueError(
            f'Incorrect index for "grid_type": {grid_type}. '
            'Supported values are 1 (uniform), 2, 3 (nonuniform).'
        )


# ---------------------------------------------------------
# Synthesis dispatcher
# ---------------------------------------------------------
def synthesize_spatial_from_fourier(u_fourier_coeff: cp.ndarray,
                                    theta_j,
                                    N: int,
                                    grid_type: int = 1,
                                    eps: float = 1e-12,
                                    **kwargs) -> cp.ndarray:
    """
    grid_type == 1: uniform IFFT on GPU
    grid_type in (2, 3): shared nonuniform, cuFINUFFT-2, theta_j (N,)
    """
    # Map legacy azu_unif alias if passed
    if "azu_unif" in kwargs:
        azu = kwargs.pop("azu_unif")
        if azu == 2:
            grid_type = 1
        elif azu in (1, 3):
            grid_type = 3

    u_fourier_coeff = cp.asarray(u_fourier_coeff)
    Np1, M = u_fourier_coeff.shape
    if Np1 != N + 1:
        raise ValueError("u_fourier_coeff must have shape (N+1, M)")

    halfN = N // 2

    if grid_type == 1:
        coeff    = cp.vstack([u_fourier_coeff[halfN:N, :],
                              u_fourier_coeff[0:halfN, :]])
        u_approx = cp.fft.ifft(coeff, axis=0) * N
        return u_approx

    elif grid_type in (2, 3):
        theta = cp.asarray(theta_j, dtype=float)
        if theta.ndim != 1 or theta.size != N:
            raise ValueError("theta_j must be 1D of length N when grid_type in (2, 3)")
        x        = cp.ascontiguousarray(_wrap_angles(theta))
        coeff    = u_fourier_coeff[:N, :].copy()
        coeff[0, :] += u_fourier_coeff[N, :]  # Recombine the split Nyquist mode (k = -N/2 and +N/2)
        coeff_KN = cp.ascontiguousarray(coeff.T, dtype=cp.complex128)  # (M, N)
        
        plan = cufinufft.Plan(2, (N,), n_trans=M, isign=+1, eps=eps, dtype=np.complex128)
        plan.setpts(x)
        out_KM = cp.empty((M, N), dtype=cp.complex128)
        plan.execute(coeff_KN, out_KM)
        return out_KM.T                                                # (N, M)

    else:
        raise ValueError(
            f'Incorrect index for "grid_type": {grid_type}. '
            'Supported values are 1 (uniform), 2, 3 (nonuniform).'
        )


@cp.fuse()
def _fuse_dirichlet_modes(v_mask, ratio, kabs, g_val, v_boundary):
    return v_mask + (ratio ** kabs) * (g_val - v_boundary)

@cp.fuse()
def _fuse_neumann_modes(v_mask, ratio, kabs, R, g_val, v_boundary):
    return v_mask + (ratio ** kabs) * ((R / kabs) * g_val + v_boundary)


def compute_u_fourier_coefficients(v: cp.ndarray,
                                   g_fourier_coeff: cp.ndarray,
                                   u_fourier_0: complex,
                                   N: int,
                                   M: int,
                                   r_m: cp.ndarray,
                                   R: float,
                                   BC_choice: int) -> cp.ndarray:
    """
    Compute u_n(r) Fourier coefficients from v_n(r) and boundary data on GPU.
    """
    halfN = N // 2
    u_fourier_coeff = cp.zeros((N + 1, M), dtype=complex)

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
        C = u_fourier_0 - v[halfN, -1]
        u_fourier_coeff[halfN, :] = v[halfN, :] + C
    else:
        raise ValueError('Incorrect index for "BC_choice"')

    # all other modes
    n_idx = cp.arange(N + 1)
    kabs_all = cp.abs(n_idx - halfN)
    mask = n_idx != halfN  # exclude central mode

    kabs_f = kabs_all[mask].astype(cp.float64)[:, None]  # (N, 1) float64
    ratio = (r_m / R)[None, :]                           # (1, M) float64

    if BC_choice == 1:
        u_fourier_coeff[mask, :] = _fuse_dirichlet_modes(
            v[mask, :], ratio, kabs_f, g_full[mask, None], v[mask, M - 1][:, None]
        )
    elif BC_choice == 2:
        u_fourier_coeff[mask, :] = _fuse_neumann_modes(
            v[mask, :], ratio, kabs_f, float(R), g_full[mask, None], v[mask, M - 1][:, None]
        )

    return u_fourier_coeff
