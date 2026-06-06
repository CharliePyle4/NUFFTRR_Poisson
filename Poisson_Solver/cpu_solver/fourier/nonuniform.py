import numpy as np
import finufft
from scipy.sparse.linalg import LinearOperator


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
# Block CG
# ---------------------------------------------------------
def _block_cg(A_op, B, tol=1e-8, maxiter=50):
    N, K = B.shape
    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()

    def _block_norm2(M):
        return np.sum(np.abs(M)**2)

    R_norm0_sq = _block_norm2(R)
    if R_norm0_sq == 0.0:
        return X
    R_norm0 = np.sqrt(R_norm0_sq)

    for _ in range(maxiter):
        AP = A_op(P)
        R_norm_sq = _block_norm2(R)
        den = np.vdot(P, AP)
        alpha = R_norm_sq / den
        X += alpha * P
        R_new = R - alpha * AP
        R_new_norm_sq = _block_norm2(R_new)
        if np.sqrt(R_new_norm_sq) / R_norm0 < tol:
            R = R_new
            break
        beta = R_new_norm_sq / R_norm_sq
        P = R_new + beta * P
        R = R_new
    return X


# ---------------------------------------------------------
# Invert NUFFT via Block CG — shared mesh (azu_unif == 1)
# One plan for all M radii simultaneously.
# ---------------------------------------------------------

def _invert_nufft_via_block_cg(theta_j, f, tol=1e-8, maxiter=50, eps=1e-6):
    theta_j = np.asarray(theta_j, dtype=float)
    x_wrapped = _wrap_angles(theta_j)
    f = np.asarray(f, dtype=np.complex128)
    N = theta_j.size

    if f.ndim == 1:
        f = f[:, None]
    N_pts, K = f.shape

    plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=K, eps=eps)

    f_KM = np.ascontiguousarray(f.T)
    B_KN = plan_adj.execute(f_KM)
    B = np.ascontiguousarray(B_KN.T)

    V_KN_buf  = np.empty((K, N), dtype=np.complex128)
    AV_KM_buf = np.empty((K, N), dtype=np.complex128)

    def AHA_block(V):
        V_KN_buf[...] = V.T
        AV_KM = plan_fwd.execute(V_KN_buf)
        AV_KM_buf[...] = AV_KM
        AHA_V_KN = plan_adj.execute(AV_KM_buf)
        return AHA_V_KN.T

    X = _block_cg(AHA_block, B, tol=tol, maxiter=maxiter)
    return X[:, 0] if K == 1 else X


# ---------------------------------------------------------
# Invert NUFFT via CG — per-radius (azu_unif == 0)
# ---------------------------------------------------------
def _invert_nufft_via_cg_perradius(theta_j, f, tol=1e-8, maxiter=50, eps=1e-6):
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
        f_col     = f[:, ell:ell+1]                     # (N, 1) for block_cg
        plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=1, eps=eps)

        f_KM = np.ascontiguousarray(f_col.T)            # (1, N)
        B_KN = plan_adj.execute(f_KM)
        B    = np.ascontiguousarray(B_KN.T)             # (N, 1)

        V_buf   = np.empty((1, N), dtype=np.complex128)
        AV_buf  = np.empty((1, N), dtype=np.complex128)

        def AHA_block(V, _pfwd=plan_fwd, _padj=plan_adj):
            V_buf[...] = V.T
            AV = _pfwd.execute(V_buf)
            AV_buf[...] = AV
            return _padj.execute(AV_buf).T

        X = _block_cg(AHA_block, B, tol=tol, maxiter=maxiter)
        core[:, ell] = X[:, 0]

    return core


# ---------------------------------------------------------
# NUDFT inversion — shared mesh (azu_unif == 1)
# One matrix, M right-hand sides. O(N^3 + MN^2).
# ---------------------------------------------------------
def _invert_nudft(theta_j, f):
    """
    theta_j : (N,)
    f       : (N,) or (N, K)
    returns : (N,) or (N, K)
    """
    theta  = np.asarray(theta_j, float)
    f      = np.asarray(f, dtype=np.complex128)
    N      = theta.size
    k      = np.arange(-N // 2, N // 2, dtype=float)
    A      = np.exp(1j * np.outer(theta, k))    # (N, N)
    AH     = A.conj().T
    Mmat   = AH @ A

    if f.ndim == 1:
        if f.size != N:
            raise ValueError("theta_j length must equal length of f")
        return np.linalg.solve(Mmat, AH @ f)
    else:
        if f.shape[0] != N:
            raise ValueError("theta_j length must equal first dim of f")
        return np.linalg.solve(Mmat, AH @ f)    # (N, K)


# ---------------------------------------------------------
# NUDFT inversion — per-radius (azu_unif == 0)
# M matrices, batched. O(MN^3) but one LAPACK call.
# ---------------------------------------------------------
def _invert_nudft_perradius(theta_j, f):
    """
    theta_j : (N, M)
    f       : (N, M)
    returns : (N, M)
    """
    theta_j = np.asarray(theta_j, dtype=float)
    f       = np.asarray(f, dtype=np.complex128)
    N, M    = theta_j.shape

    if f.shape != (N, M):
        raise ValueError(f"f must have shape ({N}, {M}), got {f.shape}")

    k      = np.arange(-N // 2, N // 2, dtype=float)           # (N,)
    A_all  = np.exp(1j * theta_j.T[:, :, None] * k[None, None, :])  # (M, N, N)
    AH_all = A_all.conj().transpose(0, 2, 1)                    # (M, N, N)
    M_all  = AH_all @ A_all                                     # (M, N, N)
    b_all  = AH_all @ f.T[:, :, None]                          # (M, N, 1)
    X_all  = np.linalg.solve(M_all, b_all)                     # (M, N, 1)

    return X_all[:, :, 0].T                                    # (N, M)



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
        coeff_core = _invert_nufft_via_block_cg(theta_j, f_values,
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
        core = _invert_nufft_via_cg_perradius(theta_j, f_values,
                                              tol=tol, maxiter=maxiter, eps=tol)

    return _pad_coeff_to_Np1(core, N)                          # (N+1, M)


# ----------------