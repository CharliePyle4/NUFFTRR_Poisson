import numpy as np
import finufft
from scipy.linalg import lstsq
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_fft
pyfftw.interfaces.cache.enable()
import os
import multiprocessing
import scipy.fft as sp_fft


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def _resolve_num_processors(num_processors: int = None) -> int:
    """Resolve num_processors to a valid positive integer, defaulting to os.cpu_count() or 1."""
    if num_processors is None or num_processors <= 0:
        return os.cpu_count() or 1
    return int(num_processors)


def _wrap_angles(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π) for FINUFFT."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _get_density_weights(theta: np.ndarray) -> np.ndarray:
    """
    Compute normalized trapezoidal weights w_j for non-uniform angles theta in [0, 2π).
    sum(w_j) = 1.0, so that A^* W_norm A ≈ I.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        th_ext = np.concatenate(([theta[-1] - 2 * np.pi], theta, [theta[0] + 2 * np.pi]))
        w = (th_ext[2:] - th_ext[:-2]) / (4.0 * np.pi)
        return w
    return np.ones_like(theta)




def _compute_fft_kde_weights(theta_j: np.ndarray,
                             oversample: int = 4,
                             bandwidth_factor: float = 1.0,
                             num_processors: int = None) -> np.ndarray:
    """
    Density compensation via FFT-accelerated KDE on the circle — O(N log N).

    Bins angles onto a fine uniform grid, convolves with a wrapped Gaussian
    kernel via FFT, then interpolates the smoothed density back to the
    original angle positions. Weights = 1/density, normalized to sum to 1.

    Strictly positive by construction (convolution of a non-negative histogram
    with a positive Gaussian kernel), preserving the SPD property of A^* W A
    required by the Block CG solver.
    """
    N = theta_j.size
    M = int(oversample * N)  # Fine uniform grid (oversampled for interpolation accuracy)
    dx = 2 * np.pi / M

    # 1. Bin angles into uniform histogram on [0, 2pi)
    bin_edges = np.linspace(0, 2 * np.pi, M + 1)
    hist, _ = np.histogram(theta_j, bins=bin_edges)
    grid_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # 2. Wrapped Gaussian kernel centered at 0
    sigma = float(bandwidth_factor) * (2 * np.pi / N)  # Bandwidth
    k = np.arange(M) * dx
    k = np.mod(k + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)

    # 3. Circular convolution via FFT -> density at each grid center
    n_threads = _resolve_num_processors(num_processors)
    density_grid = fftw_fft.irfft(
        fftw_fft.rfft(hist.astype(float), threads=n_threads, planner_effort='FFTW_ESTIMATE')
        * fftw_fft.rfft(kernel, threads=n_threads, planner_effort='FFTW_ESTIMATE'),
        n=M, threads=n_threads, planner_effort='FFTW_ESTIMATE'
    )

    # 4. Interpolate density to original angle positions (periodic)
    density = np.interp(theta_j, grid_centers, density_grid, period=2 * np.pi)

    # 5. Invert density -> weights, normalize so sum(w) = 1
    w = 1.0 / density
    w = w / np.sum(w)

    return w



def _is_matrix(a: np.ndarray) -> bool:
    """Check if array is a matrix (2D with multiple columns)."""
    a = np.asarray(a)
    return a.ndim == 2 and a.shape[1] > 1


def _pad_coeff_to_Np1(coeff_core: np.ndarray, N: int) -> np.ndarray:
    """
    Pad NUDFT/NUFFT core output (N,) or (N, K) to (N+1,) or (N+1, K).
    Duplicates k=-N/2 to k=+N/2 and halves both endpoints for symmetry.
    """
    if coeff_core.ndim == 1:
        out = np.zeros(N + 1, dtype=np.complex128)
        out[0:N] = coeff_core
        out[N]   = coeff_core[0]
        out[0]  /= 2.0
        out[N]  /= 2.0
    else:
        K = coeff_core.shape[1]
        out = np.zeros((N + 1, K), dtype=np.complex128)
        out[0:N, :] = coeff_core
        out[N,   :] = coeff_core[0, :]
        out[0,   :] /= 2.0
        out[N,   :] /= 2.0
    return out


# ---------------------------------------------------------
# NUFFT Wrappers
# ---------------------------------------------------------
def _nufft_forward(x_wrapped, fhat, eps=1e-12, num_processors: int = None):
    n_threads = _resolve_num_processors(num_processors)
    x = np.ascontiguousarray(x_wrapped, dtype=float)
    fhat = np.asarray(fhat, dtype=np.complex128)
    if fhat.ndim == 1:
        return finufft.nufft1d2(x, np.ascontiguousarray(fhat), isign=+1, eps=eps, nthreads=n_threads)
    N_modes, K = fhat.shape
    fhat_KN = np.ascontiguousarray(fhat.T, dtype=np.complex128)
    return finufft.nufft1d2(x, fhat_KN, isign=+1, eps=eps, nthreads=n_threads).T


def _nufft_adjoint(x_wrapped, f, N_modes, eps=1e-12, num_processors: int = None):
    n_threads = _resolve_num_processors(num_processors)
    x = np.ascontiguousarray(x_wrapped, dtype=float)
    f = np.asarray(f, dtype=np.complex128)
    M = x.size
    if f.ndim == 1:
        if f.size != M:
            raise ValueError("x_wrapped length must equal length of f")
        return finufft.nufft1d1(x, np.ascontiguousarray(f), n_modes=N_modes, isign=-1, eps=eps, nthreads=n_threads)
    if f.shape[0] != M:
        raise ValueError("x_wrapped length must equal first dim of f")
    f_KM = np.ascontiguousarray(f.T, dtype=np.complex128)
    return finufft.nufft1d1(x, f_KM, n_modes=N_modes, isign=-1, eps=eps, nthreads=n_threads).T


# =============================================================================
# Direct Unsquared CGLS (Paige & Saunders) — Strategy 2 (Avoids Normal Equations)
# Solves min ||A c - f||_2 directly without squaring condition numbers.
# =============================================================================
def _compute_pipe_menon_weights(theta: np.ndarray,
                                n_iter: int = 2,
                                eps: float = 1e-12,
                                num_processors: int = None) -> np.ndarray:
    """
    Pipe & Menon (1999) Iterative Sampling Density Compensation.
    Computes mathematically optimal weights W via fixed-point sinc iteration:
    w_{k+1} = w_k / (A A^H w_k).
    """
    n_threads = _resolve_num_processors(num_processors)
    x = _wrap_angles(theta)
    N = theta.size

    # Initialize with normalized Voronoi trapezoidal weights
    theta_ext = np.concatenate([[theta[-1] - 2.0*np.pi], theta, [theta[0] + 2.0*np.pi]])
    w = 0.5 * (theta_ext[2:] - theta_ext[:-2]) / (2.0 * np.pi)

    # Fast 1D single-transform Guru plans for setup (1 thread is 15x faster for single vector)
    p1 = finufft.Plan(1, (N,), n_trans=1, isign=-1, eps=eps, nthreads=1)
    p1.setpts(x)
    p2 = finufft.Plan(2, (N,), n_trans=1, isign=+1, eps=eps, nthreads=1)
    p2.setpts(x)

    c = np.empty((1, N), dtype=np.complex128)
    d = np.empty((1, N), dtype=np.complex128)
    w_arr = w.astype(np.complex128)[None, :]

    for _ in range(n_iter):
        p1.execute(w_arr, c)
        p2.execute(c, d)
        density = np.maximum(np.real(d[0, :]), 1e-12)
        w_arr[0, :] = w_arr[0, :] / density
        w_arr[0, :] = w_arr[0, :] / np.sum(w_arr[0, :].real)

    return w_arr.real[0, :]


def _invert_nufft_cgls_unsquared(theta_j,
                                 f_arr,
                                 tol=1e-10,
                                 maxiter=200,
                                 eps=1e-12,
                                 num_processors: int = None,
                                 **kwargs):
    """
    High-Performance Preconditioned Conjugate Gradient for Least Squares (PCGLS).
    Uses persistent FINUFFT Guru Plans and Pipe & Menon optimal spatial weights.
    """
    n_threads = _resolve_num_processors(num_processors)
    theta = np.asarray(theta_j, dtype=float)
    x = _wrap_angles(theta)
    N = theta.size
    is_1d = (f_arr.ndim == 1)

    if is_1d:
        f_2d = f_arr.reshape(N, 1)
    else:
        f_2d = f_arr

    N_pts, K = f_2d.shape
    f_T = np.ascontiguousarray(f_2d.T, dtype=np.complex128)  # (K, N)
    c_T = np.zeros((K, N), dtype=np.complex128)
    r_T = f_T.copy()  # Spatial residual r = f - A c

    # Compute optimal Pipe & Menon weights
    w = _compute_pipe_menon_weights(theta, n_iter=2, eps=eps, num_processors=n_threads)[None, :]  # (1, N)

    # Initialize FINUFFT Guru Plans once outside the CGLS loop
    plan1 = finufft.Plan(1, (N,), n_trans=K, isign=-1, eps=eps, nthreads=n_threads)
    plan1.setpts(x)

    plan2 = finufft.Plan(2, (N,), n_trans=K, isign=+1, eps=eps, nthreads=n_threads)
    plan2.setpts(x)

    # Pre-allocate working memory buffers to avoid any in-loop allocations
    z_T = np.empty((K, N), dtype=np.complex128)
    np.multiply(r_T, w, out=z_T)

    s_T = np.empty((K, N), dtype=np.complex128)
    plan1.execute(z_T, s_T)

    p_T = s_T.copy()
    q_T = np.empty((K, N), dtype=np.complex128)

    gamma = np.sum(s_T.real**2 + s_T.imag**2, axis=1)  # (K,)
    norm_s0 = np.sqrt(gamma) + 1e-14

    for it in range(maxiter):
        # 1. Forward step: q = A p (Type-2 NUFFT)
        plan2.execute(p_T, q_T)

        # 2. Optimal step size (without hypot/sqrt overhead)
        norm_q_sq = np.sum((q_T.real**2 + q_T.imag**2) * w, axis=1) + 1e-28  # (K,)
        alpha = (gamma / norm_q_sq)[:, None]                                 # (K, 1)

        # 3. Update Fourier coefficients & spatial residual
        c_T += alpha * p_T
        r_T -= alpha * q_T
        np.multiply(r_T, w, out=z_T)

        # 4. Adjoint step: s = A^H (W r) (Type-1 NUFFT)
        plan1.execute(z_T, s_T)
        gamma_new = np.sum(s_T.real**2 + s_T.imag**2, axis=1)               # (K,)

        rel_res = np.max(np.sqrt(gamma_new) / norm_s0)
        if rel_res < tol:
            break

        beta = (gamma_new / (gamma + 1e-28))[:, None]
        p_T = s_T + beta * p_T
        gamma = gamma_new

    c_out = c_T.T  # (N, K)
    return c_out[:, 0] if is_1d else c_out


# =============================================================================
# PREVIOUS NORMAL-EQUATIONS BLOCK CG SOLVER (COMMENTED OUT FOR REFERENCE)
# =============================================================================
def _block_cg(T_op, B, M_inv=None, tol=1e-8, maxiter=50):
    X = B.copy()
    R = B - T_op(X)

    if M_inv is not None:
        Z = M_inv(R)
    else:
        Z = R.copy()

    P = Z.copy()
    gamma = np.vdot(R, Z).real

    if gamma <= 0.0 or np.isnan(gamma):
        return X

    norm_b_sq = np.einsum('ij,ij->i', B.real, B.real) + np.einsum('ij,ij->i', B.imag, B.imag)
    norm_b_denom_sq = (np.sqrt(norm_b_sq) + 1e-14)**2
    tol2 = tol * tol

    col_res_sq = np.einsum('ij,ij->i', R.real, R.real) + np.einsum('ij,ij->i', R.imag, R.imag)
    if np.max(col_res_sq / norm_b_denom_sq) < tol2:
        return X

    for _ in range(maxiter):
        TP = T_op(P)
        delta = np.vdot(P, TP).real
        if delta <= 0 or np.isnan(delta):
            break

        alpha = gamma / delta

        X += alpha * P
        R -= alpha * TP

        col_res_sq = np.einsum('ij,ij->i', R.real, R.real) + np.einsum('ij,ij->i', R.imag, R.imag)
        if np.max(col_res_sq / norm_b_denom_sq) < tol2:
            break

        if M_inv is not None:
            Z_new = M_inv(R)
        else:
            Z_new = R.copy()

        gamma_new = np.vdot(R, Z_new).real
        if gamma_new < 1e-28:
            break

        beta = gamma_new / (gamma + 1e-14)
        P *= beta
        P += Z_new
        gamma = gamma_new

    return X


def _invert_nufft_block_cgls_shared(theta_j,
                                    f,
                                    tol=1e-8,
                                    maxiter=50,
                                    eps=1e-12,
                                    reg_param=1e-12,
                                    precond_shift=1e-3,
                                    kde_oversample=4,
                                    kde_bandwidth=1.0,
                                    num_processors: int = None,
                                    **kwargs):
    n_threads = _resolve_num_processors(num_processors)
    theta_j = np.asarray(theta_j, dtype=float)
    f_orig = np.asarray(f, dtype=np.complex128)
    N = theta_j.size

    x_wrapped = _wrap_angles(theta_j)
    if f_orig.ndim == 1:
        f_arr = f_orig[:, None]
    else:
        f_arr = f_orig
    N_pts, K = f_arr.shape

    w = _compute_fft_kde_weights(
        theta_j,
        oversample=kde_oversample,
        bandwidth_factor=kde_bandwidth,
        num_processors=n_threads
    )[:, None]

    # 1. Compute RHS: B_adj = A^H W f  (1 FINUFFT Adjoint)
    f_w = f_arr * w
    B_adj = _nufft_adjoint(x_wrapped, f_w, N_modes=N, eps=eps, num_processors=n_threads).T  # (K, N)

    # 2. Compute Toeplitz kernel from weights (1 FINUFFT Adjoint, double resolution)
    v_raw = _nufft_adjoint(x_wrapped, w.flatten(), N_modes=2*N, eps=eps, num_processors=n_threads)  # (2N,)
    v_shift = fftw_fft.ifftshift(v_raw)
    V_hat = fftw_fft.fft(v_shift, threads=n_threads, planner_effort='FFTW_ESTIMATE')[None, :]  # (1, 2N)
    # Pre-allocate aligned arrays and build FFTW plans for T_op using FFTW_ESTIMATE (eliminates hundreds of ms of plan benchmarking)
    T_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_hat = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    fft_T = pyfftw.FFTW(T_in, T_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    T_ifft_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_out = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    ifft_T = pyfftw.FFTW(T_ifft_in, T_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    # 3. Fast Toeplitz Matrix-Vector Multiplication via FFT (zero allocations in loop)
    scale_2N = 1.0 / (2.0 * N)
    def T_op(X):
        T_in[:, :N] = X
        T_in[:, N:] = 0.0
        fft_T.execute()
        np.multiply(T_hat, V_hat, out=T_ifft_in)
        ifft_T.execute()
        return (T_out[:, :N] * scale_2N) + (reg_param * X)

    # 4. Circulant Preconditioner via T. Chan's Optimal Formula
    k = np.arange(N)
    c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
    
    # Pre-shift circulant column once so M_inv requires zero in-loop array shifts
    c_chan_shift = fftw_fft.ifftshift(c_chan)
    eig_c = np.abs(fftw_fft.fft(c_chan_shift, threads=n_threads, planner_effort='FFTW_ESTIMATE')) + precond_shift
    eig_c_inv = (1.0 / eig_c)[None, :]

    M_in = pyfftw.empty_aligned((K, N), dtype='complex128')
    M_hat = pyfftw.empty_aligned((K, N), dtype='complex128')
    fft_M = pyfftw.FFTW(M_in, M_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    M_ifft_in = pyfftw.empty_aligned((K, N), dtype='complex128')
    M_out = pyfftw.empty_aligned((K, N), dtype='complex128')
    ifft_M = pyfftw.FFTW(M_ifft_in, M_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    scale_N = 1.0 / N
    def M_inv(V):
        M_in[:] = V
        fft_M.execute()
        np.multiply(M_hat, eig_c_inv, out=M_ifft_in)
        ifft_M.execute()
        return M_out * scale_N

    # 5. Solve using Block CG (Normal Equations)
    X_T = _block_cg(T_op, B_adj, M_inv=M_inv, tol=tol, maxiter=maxiter)
    X = X_T.T
    return X[:, 0] if f_orig.ndim == 1 else X



# ---------------------------------------------------------
# NUDFT inversion — shared mesh (azu_unif == 1)
# ---------------------------------------------------------
def _invert_nudft(theta_j, f, reg_param=1e-20):
    theta = np.asarray(theta_j, float)
    f = np.asarray(f, dtype=np.complex128)
    N = theta.size
    k = np.arange(-N // 2, N // 2, dtype=float)
    A = np.exp(1j * np.outer(theta, k))

    # Option B
    return lstsq(A, f, cond=reg_param)[0]


# ---------------------------------------------------------
# Fourier Coefficient Computation — shared nonuniform (azu_unif == 1)
# ---------------------------------------------------------
def compute_fourier_coeff_nonunif(f_values: np.ndarray,
                                  theta_j: np.ndarray,
                                  grid_type: int = 3,
                                  maxiter: int = 200,
                                  tol: float = 1e-10,
                                  use_nudft: bool = False,
                                  reg_param: float = 1e-12,
                                  eps: float = 1e-12,
                                  precond_shift: float = 1e-3,
                                  kde_oversample: int = 4,
                                  kde_bandwidth: float = 1.0,
                                  num_processors: int = None,
                                  **kwargs) -> np.ndarray:
    """
    Computes azimuthal Fourier coefficients on non-uniform angular mesh theta_j.
    Uses Pipe & Menon Unsquared PCGLS for NUFFT (grid_type=3) or Block-CG Toeplitz (grid_type=2)
    or regularized least-squares for NUDFT.
    """
    f_values = np.asarray(f_values)
    N = f_values.shape[0]
    if theta_j.shape[0] != N:
        raise ValueError("theta_j and f_values must have the same first dimension")

    if use_nudft:
        coeff_core = _invert_nudft(theta_j, f_values, reg_param=reg_param)
    elif grid_type == 2:
        coeff_core = _invert_nufft_block_cgls_shared(
            theta_j, f_values,
            tol=tol, maxiter=maxiter, eps=eps,
            reg_param=reg_param,
            precond_shift=precond_shift,
            kde_oversample=kde_oversample,
            kde_bandwidth=kde_bandwidth,
            num_processors=num_processors,
            **kwargs
        )
    else:
        coeff_core = _invert_nufft_cgls_unsquared(
            theta_j, f_values,
            tol=tol, maxiter=maxiter, eps=eps,
            reg_param=reg_param,
            num_processors=num_processors,
            **kwargs
        )

    return _pad_coeff_to_Np1(coeff_core, N)