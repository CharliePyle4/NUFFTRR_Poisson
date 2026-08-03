import numpy as np
import finufft
from scipy.linalg import lstsq
from scipy.sparse.linalg import LinearOperator, lsqr


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
    elif theta.ndim == 2:
        th_ext = np.vstack(([theta[-1, :] - 2 * np.pi], theta, [theta[0, :] + 2 * np.pi]))
        return (th_ext[2:, :] - th_ext[:-2, :]) / (4.0 * np.pi)
    return np.ones_like(theta)


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
# NUFFT Plan helpers for block CG
# ---------------------------------------------------------
def _make_nufft_plans(x_wrapped, N_modes, K, eps=1e-12):
    x = np.ascontiguousarray(x_wrapped, dtype=float)
    n_modes_tuple = (int(N_modes),)
    plan_fwd = finufft.Plan(2, n_modes_tuple, n_trans=K, eps=eps, isign=+1, dtype='complex128')
    plan_fwd.setpts(x)
    plan_adj = finufft.Plan(1, n_modes_tuple, n_trans=K, eps=eps, isign=-1, dtype='complex128')
    plan_adj.setpts(x)
    return plan_fwd, plan_adj


# ---------------------------------------------------------
# Block CGLS (Conjugate Gradient for Least Squares)
# ---------------------------------------------------------
def _block_cgls(A_op, AH_op, B, M_inv=None, X_init=None, tol=1e-8, maxiter=50, damp=1e-9):
    """
    Block Preconditioned Conjugate Gradient for Least Squares (PCGLS).
    Solves min ||W^{1/2}(AX - B)||_F^2 + damp^2 ||X||_F^2 for a block of vectors.
    """
    N, K = B.shape
    if X_init is not None:
        X = X_init.astype(np.complex128, copy=True)
        R = B - A_op(X)
    else:
        X = np.zeros((N, K), dtype=np.complex128)
        R = B.astype(np.complex128, copy=True)

    S = AH_op(R)

    if damp > 0:
        S -= damp**2 * X

    if M_inv is not None:
        Z = M_inv(S)
    else:
        Z = S.copy()

    P = Z.copy()
    gamma = np.sum(np.real(np.conj(S) * Z))

    if gamma <= 0.0 or np.isnan(gamma):
        return X

    norm_b_cols = np.sqrt(np.sum(np.abs(B)**2, axis=0))
    col_residuals = np.sqrt(np.sum(np.abs(R)**2, axis=0)) / (norm_b_cols + 1e-14)
    if np.max(col_residuals) < tol:
        return X

    for _ in range(maxiter):
        Q = A_op(P)
        T_P = AH_op(Q)

        if damp > 0:
            T_P += damp**2 * P

        delta = np.sum(np.real(np.conj(P) * T_P))
        if delta <= 0 or np.isnan(delta):
            break

        alpha = gamma / delta

        X += alpha * P
        R -= alpha * Q
        S -= alpha * T_P

        col_residuals = np.sqrt(np.sum(np.abs(R)**2, axis=0)) / (norm_b_cols + 1e-14)
        if np.max(col_residuals) < tol:
            break

        if M_inv is not None:
            Z_new = M_inv(S)
        else:
            Z_new = S.copy()

        gamma_new = np.sum(np.real(np.conj(S) * Z_new))

        if gamma_new < 1e-28:
            break

        # Check relative stall (plateau detection)
        if abs(gamma_new - gamma) / (gamma + 1e-14) < 1e-6:
            break

        beta = gamma_new / (gamma + 1e-14)
        P = Z_new + beta * P
        gamma = gamma_new

    return X


# ---------------------------------------------------------
# Invert NUFFT via Block CGLS — shared mesh (azu_unif == 1)
# One plan for all M radii simultaneously with density compensation weighting.
# ---------------------------------------------------------
REG_PARAM = 1e-12  # Tikhonov regularization parameter / condition threshold


def _invert_nufft_block_cgls_shared(theta_j, f, tol=1e-8, maxiter=50, eps=1e-7):
    theta_j = np.asarray(theta_j, dtype=float)
    f_orig = np.asarray(f, dtype=np.complex128)
    N = theta_j.size

    x_wrapped = _wrap_angles(theta_j)
    if f_orig.ndim == 1:
        f_arr = f_orig[:, None]
    else:
        f_arr = f_orig
    N_pts, K = f_arr.shape

    w = _get_density_weights(theta_j)[:, None]  # (N, 1)

    plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=K, eps=eps)

    fwd_in_buf = np.empty((K, N), dtype=np.complex128)
    fwd_out_buf = np.empty((K, N_pts), dtype=np.complex128)
    adj_in_buf = np.empty((K, N_pts), dtype=np.complex128)
    adj_out_buf = np.empty((K, N), dtype=np.complex128)

    def A_op(C_block):
        fwd_in_buf[...] = C_block.T
        plan_fwd.execute(fwd_in_buf, out=fwd_out_buf)
        return fwd_out_buf.T

    def AH_op(D_block):
        adj_in_buf[...] = (D_block * w).T
        plan_adj.execute(adj_in_buf, out=adj_out_buf)
        return adj_out_buf.T

    # 1. Circulant Preconditioner via Point Spread Function (PSF)
    ones_vec = np.ones((N_pts, 1), dtype=np.complex128)
    c_psf = AH_op(ones_vec)[:, 0]
    c_psf_fft = np.fft.ifftshift(c_psf)
    eig_c = np.abs(np.fft.fft(c_psf_fft)) + 1e-3

    def M_inv(V):
        V_shift = np.fft.ifftshift(V, axes=0)
        V_hat = np.fft.fft(V_shift, axis=0)
        V_prec_hat = V_hat / eig_c[:, None]
        return np.fft.fftshift(np.fft.ifft(V_prec_hat, axis=0), axes=0)

    # 2. Density-compensated initial guess (Warm-Start)
    X0 = AH_op(f_arr)

    # Block PCGLS with FINUFFT operators
    X = _block_cgls(A_op, AH_op, f_arr, M_inv=M_inv, X_init=X0, tol=tol, maxiter=maxiter, damp=REG_PARAM)
    return X[:, 0] if f_orig.ndim == 1 else X


# ---------------------------------------------------------
# Invert NUFFT via LSQR — per-radius (azu_unif == 0)
# ---------------------------------------------------------
def _invert_nufft_lsqr_perradius(theta_j, f, tol=1e-8, maxiter=50, eps=1e-6):
    theta_j = np.asarray(theta_j, dtype=float)
    f       = np.asarray(f, dtype=np.complex128)
    N, M    = theta_j.shape

    core = np.zeros((N, M), dtype=np.complex128)
    for ell in range(M):
        x_wrapped = _wrap_angles(theta_j[:, ell])
        w = _get_density_weights(theta_j[:, ell])
        w_sqrt = np.sqrt(w)

        plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=1, eps=eps)

        def _matvec(c, _pfwd=plan_fwd, _wsqrt=w_sqrt):
            c_buf = np.ascontiguousarray(c[None, :])
            return _pfwd.execute(c_buf)[0, :] * _wsqrt

        def _rmatvec(d, _padj=plan_adj, _wsqrt=w_sqrt):
            d_buf = np.ascontiguousarray((d * _wsqrt)[None, :])
            return _padj.execute(d_buf)[0, :]

        A_op = LinearOperator(shape=(N, N), matvec=_matvec, rmatvec=_rmatvec, dtype=np.complex128)
        f_w = f[:, ell] * w_sqrt

        core[:, ell] = lsqr(A_op, f_w, damp=REG_PARAM, iter_lim=maxiter, atol=tol, btol=tol)[0]

    return core


# ---------------------------------------------------------
# NUDFT inversion — shared mesh (azu_unif == 1)
# ---------------------------------------------------------
def _invert_nudft(theta_j, f):
    theta = np.asarray(theta_j, float)
    f = np.asarray(f, dtype=np.complex128)
    N = theta.size
    k = np.arange(-N // 2, N // 2, dtype=float)
    A = np.exp(1j * np.outer(theta, k))

    w = _get_density_weights(theta)
    w_sqrt = np.sqrt(w)
    A_w = A * w_sqrt[:, None]

    if f.ndim == 1:
        f_w = f * w_sqrt
    else:
        f_w = f * w_sqrt[:, None]

    return lstsq(A_w, f_w, cond=REG_PARAM)[0]


# ---------------------------------------------------------
# NUDFT inversion — per-radius (azu_unif == 0)
# ---------------------------------------------------------
def _invert_nudft_perradius(theta_j, f):
    theta_j = np.asarray(theta_j, dtype=float)
    f = np.asarray(f, dtype=np.complex128)
    N, M = theta_j.shape

    if f.shape != (N, M):
        raise ValueError(f"f must have shape ({N}, {M}), got {f.shape}")

    k = np.arange(-N // 2, N // 2, dtype=float)
    # 3D Tensor of Fourier matrices for all M radii: shape (M, N, N)
    A_all = np.exp(1j * theta_j.T[:, :, None] * k[None, None, :])

    w = _get_density_weights(theta_j)   # shape (N, M)
    w_sqrt_T = np.sqrt(w).T             # shape (M, N)

    A_w = A_all * w_sqrt_T[:, :, None]  # shape (M, N, N)
    f_w = f.T * w_sqrt_T                # shape (M, N)

    # Solves all M linear systems simultaneously using batched numpy lstsq
    X_all = np.linalg.lstsq(A_w, f_w[:, :, None], rcond=REG_PARAM)[0]
    return X_all.squeeze(-1).T           # shape (N, M)


# ---------------------------------------------------------
# Fourier Coefficient Computation — shared nonuniform (azu_unif == 1)
# ---------------------------------------------------------
def compute_fourier_coeff_nonunif(f_values: np.ndarray,
                                  theta_j: np.ndarray,
                                  maxiter: int = 50,
                                  tol: float = 1e-8,
                                  use_nudft: bool = False) -> np.ndarray:
    """
    theta_j : (N,)       — same mesh for all radii
    f_values: (N,) or (N, M)
    """
    f_values = np.asarray(f_values)
    N = f_values.shape[0]
    if theta_j.shape[0] != N:
        raise ValueError("theta_j and f_values must have the same first dimension")

    if use_nudft:
        coeff_core = _invert_nudft(theta_j, f_values)
    else:
        coeff_core = _invert_nufft_block_cgls_shared(theta_j, f_values,
                                                     tol=tol, maxiter=maxiter, eps=1e-7)

    return _pad_coeff_to_Np1(coeff_core, N)


# ---------------------------------------------------------
# Fourier Coefficient Computation — per-radius nonuniform (azu_unif == 0)
# ---------------------------------------------------------
def compute_fourier_coeff_nonunif_perradius(f_values: np.ndarray,
                                            theta_j: np.ndarray,
                                            maxiter: int = 200,
                                            tol: float = 1e-8,
                                            use_nudft: bool = True) -> np.ndarray:
    """
    theta_j : (N, M)     — different mesh per radius
    f_values: (N, M)
    """
    f_values = np.asarray(f_values, dtype=np.complex128)
    theta_j  = np.asarray(theta_j, dtype=float)
    N, M     = f_values.shape

    if theta_j.shape != (N, M):
        raise ValueError(f"theta_j must have shape ({N}, {M}), got {theta_j.shape}")

    if use_nudft:
        core = _invert_nudft_perradius(theta_j, f_values)      # (N, M)
    else:
        core = _invert_nufft_lsqr_perradius(theta_j, f_values,
                                             tol=tol, maxiter=maxiter, eps=tol)

    return _pad_coeff_to_Np1(core, N)                          # (N+1, M)