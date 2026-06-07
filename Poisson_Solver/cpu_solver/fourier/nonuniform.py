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


def _is_matrix(a: np.ndarray) -> bool:
    """Check if array is a matrix (2D with multiple columns)."""
    a = np.asarray(a)
    return a.ndim == 2 and a.shape[1] > 1


def _pad_coeff_to_Np1(coeff_core: np.ndarray, N: int) -> np.ndarray:
    """
    Pad NUDFT/NUFFT core output (N,) or (N, K) to (N+1,) or (N+1, K).
    Duplicates k=-N/2 to k=+N/2 and halves both endpoints.
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
def _block_cgls(A_op, AH_op, B, tol=1e-8, maxiter=50, damp=1e-9):
    """
    Block Conjugate Gradient for Least Squares (CGLS).
    Solves min ||AX - B||_F^2 + damp^2 ||X||_F^2 for a block of vectors.
    A_op: function for matrix-block product A @ X
    AH_op: function for adjoint matrix-block product A.H @ X
    B: block of right-hand sides (N, K)
    """
    N, K = B.shape
    X = np.zeros_like(B, dtype=np.complex128)
    R = B.copy()
    S = AH_op(R)

    # Add damping term to the gradient for the regularized problem
    if damp > 0:
        S -= damp**2 * X

    P = S.copy()
    gamma = np.sum(np.abs(S)**2)

    if gamma == 0.0:
        return X

    norm_b_sq = np.sum(np.abs(B)**2)

    for _ in range(maxiter):
        Q = A_op(P)

        # Add damping term for regularization
        if damp > 0:
            Q += damp * P

        delta = np.sum(np.abs(Q)**2)
        if delta == 0.0:
            # This can happen if P is in the null space of the augmented operator
            break

        alpha = gamma / delta

        X += alpha * P
        R -= alpha * Q
        S_new = AH_op(R)

        if damp > 0:
            S_new -= damp**2 * X

        gamma_new = np.sum(np.abs(S_new)**2)

        # Convergence check using the norm of the residual of the normal equations
        if np.sqrt(gamma_new) < tol * np.sqrt(norm_b_sq):
            break

        beta = gamma_new / gamma
        P = S_new + beta * P
        gamma = gamma_new

    return X


# ---------------------------------------------------------
# Invert NUFFT via Block CGLS — shared mesh (azu_unif == 1)
# One plan for all M radii simultaneously.
# ---------------------------------------------------------
REG_PARAM = 1e-9  # Tikhonov regularization parameter


def _invert_nufft_block_cgls_shared(theta_j, f, tol=1e-8, maxiter=50, eps=1e-6):
    theta_j = np.asarray(theta_j, dtype=float)
    x_wrapped = _wrap_angles(theta_j)
    f = np.asarray(f, dtype=np.complex128)
    N = theta_j.size

    if f.ndim == 1:
        f = f[:, None]
    N_pts, K = f.shape # N_pts is also N here

    # Create batched plans for all K transforms at once
    plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=K, eps=eps)

    # Buffers for contiguous memory access to be used with plan.execute
    fwd_in_buf = np.empty((K, N), dtype=np.complex128)
    fwd_out_buf = np.empty((K, N_pts), dtype=np.complex128)
    adj_in_buf = np.empty((K, N_pts), dtype=np.complex128)

    def A_op(C_block): # Applies A to a block of columns
        fwd_in_buf[...] = C_block.T
        plan_fwd.execute(fwd_in_buf, out=fwd_out_buf)
        return fwd_out_buf.T

    def AH_op(D_block): # Applies A.H to a block of columns
        adj_in_buf[...] = D_block.T
        return plan_adj.execute(adj_in_buf).T

    # Solve min ||AX - F||^2 using block CGLS
    X = _block_cgls(A_op, AH_op, f, tol=tol, maxiter=maxiter, damp=REG_PARAM)

    return X[:, 0] if K == 1 else X


# ---------------------------------------------------------
# Invert NUFFT via LSQR — per-radius (azu_unif == 0)
# ---------------------------------------------------------
def _invert_nufft_lsqr_perradius(theta_j, f, tol=1e-8, maxiter=50, eps=1e-6):
    """
    theta_j : (N, M)
    f       : (N, M)
    returns : (N, M)
    """
    theta_j = np.asarray(theta_j, dtype=float)
    f       = np.asarray(f, dtype=np.complex128)
    N, M    = theta_j.shape

    core = np.zeros((N, M), dtype=np.complex128)
    for ell in range(M):
        x_wrapped = _wrap_angles(theta_j[:, ell])

        # A new operator must be defined for each radius, as the points change.
        plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=1, eps=eps)

        def _matvec(c, _pfwd=plan_fwd):
            c_buf = np.ascontiguousarray(c[None, :])
            return _pfwd.execute(c_buf)[0, :]

        def _rmatvec(d, _padj=plan_adj):
            d_buf = np.ascontiguousarray(d[None, :])
            return _padj.execute(d_buf)[0, :]

        A_op = LinearOperator(shape=(N, N), matvec=_matvec, rmatvec=_rmatvec, dtype=np.complex128)

        # lsqr is a fast, compiled routine for solving least-squares problems.
        core[:, ell] = lsqr(A_op, f[:, ell], damp=REG_PARAM, iter_lim=maxiter, atol=tol, btol=tol)[0]

    return core


# ---------------------------------------------------------
# NUDFT inversion — shared mesh (azu_unif == 1)
# Solves min ||Ax - f|| directly using scipy.linalg.lstsq for stability.
# ---------------------------------------------------------
def _invert_nudft(theta_j, f):
    """
    theta_j : (N,)
    f       : (N,) or (N, K)
    returns : (N,) or (N, K)
    """
    theta = np.asarray(theta_j, float)
    f = np.asarray(f, dtype=np.complex128)
    N = theta.size
    k = np.arange(-N // 2, N // 2, dtype=float)
    A = np.exp(1j * np.outer(theta, k))  # (N, N)

    # Use lstsq for numerical stability. rcond acts as a regularizer.
    # It handles both vector and matrix f.
    return lstsq(A, f, cond=REG_PARAM)[0]


# ---------------------------------------------------------
# NUDFT inversion — per-radius (azu_unif == 0)
# M matrices, batched. O(MN^3) using batched least-squares.
# ---------------------------------------------------------
def _invert_nudft_perradius(theta_j, f):
    """
    theta_j : (N, M)
    f       : (N, M)
    returns : (N, M)
    """
    theta_j = np.asarray(theta_j, dtype=float)
    f = np.asarray(f, dtype=np.complex128)
    N, M = theta_j.shape

    if f.shape != (N, M):
        raise ValueError(f"f must have shape ({N}, {M}), got {f.shape}")

    k = np.arange(-N // 2, N // 2, dtype=float)
    # A_all has shape (M, N, N)
    A_all = np.exp(1j * theta_j.T[:, :, None] * k[None, None, :])
    # f.T has shape (M, N). np.linalg.lstsq solves M systems in a batch.
    X_all = np.linalg.lstsq(A_all, f.T, rcond=REG_PARAM)[0]
    # Transpose result from (M, N) back to (N, M)
    return X_all.T


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
                                                     tol=tol, maxiter=maxiter, eps=tol)

    return _pad_coeff_to_Np1(coeff_core, N)


# ---------------------------------------------------------
# Fourier Coefficient Computation — per-radius nonuniform (azu_unif == 0)
# ---------------------------------------------------------
def compute_fourier_coeff_nonunif_perradius(f_values: np.ndarray,
                                            theta_j: np.ndarray,
                                            maxiter: int = 50,
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