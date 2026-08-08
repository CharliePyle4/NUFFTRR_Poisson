import numpy as np
import finufft
from scipy.linalg import lstsq
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_fft
pyfftw.interfaces.cache.enable()
import multiprocessing
import scipy.fft as sp_fft


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
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


'''def _compute_iterative_dcf(theta_j: np.ndarray, N_modes: int, iters: int = 15, eps: float = 1e-12) -> np.ndarray:
    """
    Computes mathematically optimal density compensation weights iteratively (Pipe-Menon algorithm).
    Forces A A^* W = I to perfectly condition the Fourier matrix, eliminating geometric noise.
    """
    x_wrapped = _wrap_angles(theta_j)
    w = _get_density_weights(theta_j)  # Use geometric weights as a good initial guess
    
    for _ in range(iters):
        # 1. Spatial -> Modes (Adjoint NUFFT)
        u = _nufft_adjoint(x_wrapped, w, N_modes=N_modes, eps=eps)
        # 2. Modes -> Spatial (Forward NUFFT)
        v = _nufft_forward(x_wrapped, u, eps=eps)
        # 3. Update weights
        w = w / (v.real + 1e-14)
        
    return w'''

def _compute_fft_kde_weights(theta_j: np.ndarray) -> np.ndarray:
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
    M = 4 * N  # Fine uniform grid (4x oversampled for interpolation accuracy)
    dx = 2 * np.pi / M

    # 1. Bin angles into uniform histogram on [0, 2pi)
    bin_edges = np.linspace(0, 2 * np.pi, M + 1)
    hist, _ = np.histogram(theta_j, bins=bin_edges)
    grid_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # 2. Wrapped Gaussian kernel centered at 0
    sigma = 2 * np.pi / N  # Bandwidth = average angular spacing
    k = np.arange(M) * dx
    k = np.mod(k + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)

    # 3. Circular convolution via FFT -> density at each grid center
    n_threads = multiprocessing.cpu_count()
    density_grid = fftw_fft.irfft(
        fftw_fft.rfft(hist.astype(float), threads=n_threads)
        * fftw_fft.rfft(kernel, threads=n_threads),
        n=M, threads=n_threads
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
def _nufft_forward(x_wrapped, fhat, eps=1e-12):
    x = np.ascontiguousarray(x_wrapped, dtype=float)
    fhat = np.asarray(fhat, dtype=np.complex128)
    if fhat.ndim == 1:
        return finufft.nufft1d2(x, np.ascontiguousarray(fhat), isign=+1, eps=eps)
    N_modes, K = fhat.shape
    fhat_KN = np.ascontiguousarray(fhat.T, dtype=np.complex128)
    return finufft.nufft1d2(x, fhat_KN, isign=+1, eps=eps).T


def _nufft_adjoint(x_wrapped, f, N_modes, eps=1e-12):
    x = np.ascontiguousarray(x_wrapped, dtype=float)
    f = np.asarray(f, dtype=np.complex128)
    M = x.size
    if f.ndim == 1:
        if f.size != M:
            raise ValueError("x_wrapped length must equal length of f")
        return finufft.nufft1d1(x, np.ascontiguousarray(f), n_modes=N_modes, isign=-1, eps=eps)
    if f.shape[0] != M:
        raise ValueError("x_wrapped length must equal first dim of f")
    f_KM = np.ascontiguousarray(f.T, dtype=np.complex128)
    return finufft.nufft1d1(x, f_KM, n_modes=N_modes, isign=-1, eps=eps).T


# ---------------------------------------------------------
# Block CG (Conjugate Gradient for Normal Equations)
# ---------------------------------------------------------
def _block_cg(T_op, B, M_inv=None, tol=1e-8, maxiter=50):
    """
    Block Conjugate Gradient for solving symmetric positive definite systems T * X = B.
    B has shape (K, N).
    """
    X = np.zeros_like(B)
    R = B.copy()

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

        if abs(gamma_new - gamma) / (gamma + 1e-14) < 1e-6:
            break

        beta = gamma_new / (gamma + 1e-14)
        P *= beta
        P += Z_new
        gamma = gamma_new

    return X


# ---------------------------------------------------------
# Invert NUFFT via Block CG & Toeplitz Embedding — shared mesh (azu_unif == 1)
# ---------------------------------------------------------
def _invert_nufft_block_cgls_shared(theta_j, f, tol=1e-8, maxiter=50, eps=1e-9, reg_param=1e-12):
    theta_j = np.asarray(theta_j, dtype=float)
    f_orig = np.asarray(f, dtype=np.complex128)
    N = theta_j.size

    x_wrapped = _wrap_angles(theta_j)
    if f_orig.ndim == 1:
        f_arr = f_orig[:, None]
    else:
        f_arr = f_orig
    N_pts, K = f_arr.shape

    
    
    #w = _compute_iterative_dcf(theta_j, N, iters=15, eps=eps)[:, None]  # Optimal iterative weights
    w = _compute_fft_kde_weights(theta_j)[:, None]  # FFT-KDE weights (O(N log N), strictly positive)

    # 1. Compute RHS: B_adj = A^H W f  (1 FINUFFT Adjoint)
    f_w = f_arr * w
    B_adj = _nufft_adjoint(x_wrapped, f_w, N_modes=N, eps=eps).T  # (K, N)

    # 2. Compute Toeplitz kernel from weights (1 FINUFFT Adjoint, double resolution)
    n_threads = multiprocessing.cpu_count()

    v_raw = _nufft_adjoint(x_wrapped, w.flatten(), N_modes=2*N, eps=eps)  # (2N,)
    v_shift = fftw_fft.ifftshift(v_raw)
    V_hat = fftw_fft.fft(v_shift, threads=n_threads)[None, :]  # (1, 2N)

    # Pre-allocate aligned arrays and build FFTW plans for T_op
    T_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_hat = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    fft_T = pyfftw.FFTW(T_in, T_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads)

    T_ifft_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_out = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    ifft_T = pyfftw.FFTW(T_ifft_in, T_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads)

    # 3. Fast Toeplitz Matrix-Vector Multiplication via FFT
    def T_op(X):
        # X: (K, N)
        T_in[:] = 0.0
        T_in[:, :N] = X
        fft_T.execute()
        T_ifft_in[:] = T_hat * V_hat
        ifft_T.execute()
        return (T_out[:, :N].copy() / (2.0 * N)) + (reg_param**2) * X

    # 4. Circulant Preconditioner via T. Chan's Optimal Formula
    # FINUFFT modes run from -N to N-1, so k=0 is at index N.
    k = np.arange(N)
    # Weighted average of the diagonals of the Toeplitz matrix
    c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
    
    eig_c = np.abs(fftw_fft.fft(c_chan, threads=n_threads)) + 1e-3
    eig_c_inv = (1.0 / eig_c)[None, :]

    M_in = pyfftw.empty_aligned((K, N), dtype='complex128')
    M_hat = pyfftw.empty_aligned((K, N), dtype='complex128')
    fft_M = pyfftw.FFTW(M_in, M_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads)

    M_ifft_in = pyfftw.empty_aligned((K, N), dtype='complex128')
    M_out = pyfftw.empty_aligned((K, N), dtype='complex128')
    ifft_M = pyfftw.FFTW(M_ifft_in, M_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads)

    def M_inv(V):
        M_in[:] = fftw_fft.ifftshift(V, axes=1)
        fft_M.execute()
        M_ifft_in[:] = M_hat * eig_c_inv
        ifft_M.execute()
        return fftw_fft.fftshift(M_out / N, axes=1).copy()

    # 5. Solve using Block CG (Normal Equations)
    X_T = _block_cg(T_op, B_adj, M_inv=M_inv, tol=tol, maxiter=maxiter)
    X = X_T.T
    return X[:, 0] if f_orig.ndim == 1 else X


# ---------------------------------------------------------
# NUDFT inversion — shared mesh (azu_unif == 1)
# ---------------------------------------------------------
def _invert_nudft(theta_j, f, reg_param=1e-12):
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
                                  maxiter: int = 50,
                                  tol: float = 1e-8,
                                  use_nudft: bool = False,
                                  reg_param: float = 1e-12) -> np.ndarray:
    """
    theta_j : (N,)       — same mesh for all radii
    f_values: (N,) or (N, M)
    """
    f_values = np.asarray(f_values)
    N = f_values.shape[0]
    if theta_j.shape[0] != N:
        raise ValueError("theta_j and f_values must have the same first dimension")

    if use_nudft:
        coeff_core = _invert_nudft(theta_j, f_values, reg_param=reg_param)
    else:
        coeff_core = _invert_nufft_block_cgls_shared(theta_j, f_values,
                                                     tol=tol, maxiter=maxiter, eps=1e-12,
                                                     reg_param=reg_param)

    return _pad_coeff_to_Np1(coeff_core, N)