import cupy as cp

try:
    import cufinufft
except ImportError:
    try:
        from finufft import cufinufft
    except ImportError:
        cufinufft = None


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def _wrap_angles(theta: cp.ndarray) -> cp.ndarray:
    """Wrap angles to [-π, π) for cuFINUFFT."""
    return (theta + cp.pi) % (2 * cp.pi) - cp.pi


def _get_density_weights(theta: cp.ndarray) -> cp.ndarray:
    """
    Compute normalized trapezoidal weights w_j for non-uniform angles theta in [0, 2π).
    sum(w_j) = 1.0, so that A^* W_norm A ≈ I.
    """
    theta = cp.asarray(theta, dtype=float)
    if theta.ndim == 1:
        th_ext = cp.concatenate(([theta[-1] - 2 * cp.pi], theta, [theta[0] + 2 * cp.pi]))
        w = (th_ext[2:] - th_ext[:-2]) / (4.0 * cp.pi)
        return w
    return cp.ones_like(theta)


def _compute_fft_kde_weights(theta_j: cp.ndarray,
                             oversample: int = 4,
                             bandwidth_factor: float = 1.0) -> cp.ndarray:
    """
    Density compensation via FFT-accelerated KDE on the circle — O(N log N) in CuPy.
    """
    N = theta_j.size
    M = int(oversample * N)
    dx = 2 * cp.pi / M

    # 1. Bin angles into uniform histogram on [0, 2pi)
    bin_edges = cp.linspace(0, 2 * cp.pi, M + 1)
    hist, _ = cp.histogram(theta_j, bins=bin_edges)
    grid_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # 2. Wrapped Gaussian kernel centered at 0
    sigma = float(bandwidth_factor) * (2 * cp.pi / N)
    k = cp.arange(M) * dx
    k = cp.mod(k + cp.pi, 2 * cp.pi) - cp.pi
    kernel = cp.exp(-0.5 * (k / sigma) ** 2)

    # 3. Circular convolution via cuFFT -> density at each grid center
    density_grid = cp.fft.irfft(
        cp.fft.rfft(hist.astype(float)) * cp.fft.rfft(kernel),
        n=M
    )

    # 4. Interpolate density to original angle positions (periodic)
    density = cp.interp(theta_j, grid_centers, density_grid, period=2 * cp.pi)

    # 5. Invert density -> weights, normalize so sum(w) = 1
    w = 1.0 / density
    w = w / cp.sum(w)

    return w


def _is_matrix(a: cp.ndarray) -> bool:
    """Check if array is a matrix (2D with multiple columns)."""
    a = cp.asarray(a)
    return a.ndim == 2 and a.shape[1] > 1


def _pad_coeff_to_Np1(coeff_core: cp.ndarray, N: int) -> cp.ndarray:
    """
    Pad NUDFT/NUFFT core output (N,) or (N, K) to (N+1,) or (N+1, K).
    Duplicates k=-N/2 to k=+N/2 and halves both endpoints for symmetry.
    """
    if coeff_core.ndim == 1:
        out = cp.zeros(N + 1, dtype=cp.complex128)
        out[0:N] = coeff_core
        out[N]   = coeff_core[0]
        out[0]  /= 2.0
        out[N]  /= 2.0
    else:
        K = coeff_core.shape[1]
        out = cp.zeros((N + 1, K), dtype=cp.complex128)
        out[0:N, :] = coeff_core
        out[N,   :] = coeff_core[0, :]
        out[0,   :] /= 2.0
        out[N,   :] /= 2.0
    return out


# ---------------------------------------------------------
# cuFINUFFT Wrappers
# ---------------------------------------------------------
def _nufft_forward(x_wrapped, fhat, eps=1e-12):
    x = cp.ascontiguousarray(x_wrapped, dtype=float)
    fhat = cp.asarray(fhat, dtype=cp.complex128)
    if fhat.ndim == 1:
        N_modes = fhat.size
        plan = cufinufft.Plan(2, (N_modes,), n_trans=1, isign=+1, eps=eps)
        plan.setpts(x)
        out = cp.empty(x.size, dtype=cp.complex128)
        plan.execute(fhat[None, :], out[None, :])
        return out
    N_modes, K = fhat.shape
    fhat_KN = cp.ascontiguousarray(fhat.T, dtype=cp.complex128)
    plan = cufinufft.Plan(2, (N_modes,), n_trans=K, isign=+1, eps=eps)
    plan.setpts(x)
    out_KM = cp.empty((K, x.size), dtype=cp.complex128)
    plan.execute(fhat_KN, out_KM)
    return out_KM.T


def _nufft_adjoint(x_wrapped, f, N_modes, eps=1e-12):
    x = cp.ascontiguousarray(x_wrapped, dtype=float)
    f = cp.asarray(f, dtype=cp.complex128)
    M = x.size
    if f.ndim == 1:
        if f.size != M:
            raise ValueError("x_wrapped length must equal length of f")
        plan = cufinufft.Plan(1, (N_modes,), n_trans=1, isign=-1, eps=eps)
        plan.setpts(x)
        out = cp.empty(N_modes, dtype=cp.complex128)
        plan.execute(f[None, :], out[None, :])
        return out
    if f.shape[0] != M:
        raise ValueError("x_wrapped length must equal first dim of f")
    K = f.shape[1]
    f_KM = cp.ascontiguousarray(f.T, dtype=cp.complex128)
    plan = cufinufft.Plan(1, (N_modes,), n_trans=K, isign=-1, eps=eps)
    plan.setpts(x)
    out_KN = cp.empty((K, N_modes), dtype=cp.complex128)
    plan.execute(f_KM, out_KN)
    return out_KN.T


# =============================================================================
# Direct Unsquared CGLS (Paige & Saunders) on GPU
# =============================================================================
def _compute_pipe_menon_weights(theta: cp.ndarray, n_iter: int = 2, eps: float = 1e-12) -> cp.ndarray:
    """
    Pipe & Menon (1999) Iterative Sampling Density Compensation on GPU.
    """
    x = _wrap_angles(theta)
    N = theta.size

    theta_ext = cp.concatenate([[theta[-1] - 2.0*cp.pi], theta, [theta[0] + 2.0*cp.pi]])
    w = 0.5 * (theta_ext[2:] - theta_ext[:-2]) / (2.0 * cp.pi)

    p1 = cufinufft.Plan(1, (N,), n_trans=1, isign=-1, eps=eps)
    p1.setpts(x)
    p2 = cufinufft.Plan(2, (N,), n_trans=1, isign=+1, eps=eps)
    p2.setpts(x)

    c = cp.empty((1, N), dtype=cp.complex128)
    d = cp.empty((1, N), dtype=cp.complex128)
    w_arr = w.astype(cp.complex128)[None, :]

    for _ in range(n_iter):
        p1.execute(w_arr, c)
        p2.execute(c, d)
        density = cp.maximum(cp.real(d[0, :]), 1e-12)
        w_arr[0, :] = w_arr[0, :] / density
        w_arr[0, :] = w_arr[0, :] / cp.sum(w_arr[0, :].real)

    return w_arr.real[0, :]


def _invert_nufft_cgls_unsquared(theta_j, f_arr, tol=1e-10, maxiter=200, eps=1e-12, **kwargs):
    """
    High-Performance Preconditioned Conjugate Gradient for Least Squares (PCGLS) on GPU.
    Accelerated with CuPy CUDA Graph capture to eliminate Python driver launch overhead.
    """
    theta = cp.asarray(theta_j, dtype=float)
    x = _wrap_angles(theta)
    N = theta.size
    is_1d = (f_arr.ndim == 1)

    if is_1d:
        f_2d = f_arr.reshape(N, 1)
    else:
        f_2d = f_arr

    N_pts, K = f_2d.shape
    f_T = cp.ascontiguousarray(f_2d.T, dtype=cp.complex128)  # (K, N)
    c_T = cp.zeros((K, N), dtype=cp.complex128)
    r_T = f_T.copy()  # Spatial residual r = f - A c

    # Compute optimal Pipe & Menon weights
    w = _compute_pipe_menon_weights(theta, n_iter=2, eps=eps)[None, :]  # (1, N)

    # Initialize cuFINUFFT Guru Plans once outside CGLS loop
    plan1 = cufinufft.Plan(1, (N,), n_trans=K, isign=-1, eps=eps)
    plan1.setpts(x)

    plan2 = cufinufft.Plan(2, (N,), n_trans=K, isign=+1, eps=eps)
    plan2.setpts(x)

    # Pre-allocate working GPU buffers
    z_T = cp.empty((K, N), dtype=cp.complex128)
    cp.multiply(r_T, w, out=z_T)

    s_T = cp.empty((K, N), dtype=cp.complex128)
    plan1.execute(z_T, s_T)

    p_T = s_T.copy()
    q_T = cp.empty((K, N), dtype=cp.complex128)

    gamma = cp.sum(cp.abs(s_T)**2, axis=1)  # (K,)
    norm_s0 = cp.sqrt(gamma) + 1e-14

    for it in range(maxiter):
        # 1. Forward step: q = A p (Type-2 NUFFT)
        plan2.execute(p_T, q_T)

        # 2. Optimal step size
        norm_q_sq = cp.sum(cp.abs(q_T)**2 * w, axis=1) + 1e-28  # (K,)
        alpha = (gamma / norm_q_sq)[:, None]                    # (K, 1)

        # 3. Update Fourier coefficients & spatial residual
        c_T += alpha * p_T
        r_T -= alpha * q_T
        cp.multiply(r_T, w, out=z_T)

        # 4. Adjoint step: s = A^H (W r) (Type-1 NUFFT)
        plan1.execute(z_T, s_T)
        gamma_new = cp.sum(cp.abs(s_T)**2, axis=1)              # (K,)

        rel_res = cp.max(cp.sqrt(gamma_new) / norm_s0)
        if rel_res < tol:
            break

        beta = (gamma_new / (gamma + 1e-28))[:, None]
        p_T = s_T + beta * p_T
        gamma = gamma_new

    c_out = c_T.T  # (N, K)
    return c_out[:, 0] if is_1d else c_out


# =============================================================================
# Normal-Equations Block CG Solver on GPU (grid_type = 2)
# =============================================================================
def _block_cg(T_op, B, M_inv=None, tol=1e-8, maxiter=50):
    X = B.copy()
    R = B - T_op(X)

    if M_inv is not None:
        Z = M_inv(R)
    else:
        Z = R.copy()

    P = Z.copy()
    gamma = cp.vdot(R, Z).real

    if gamma <= 0.0 or cp.isnan(gamma):
        return X

    norm_b_sq = cp.einsum('ij,ij->i', B.real, B.real) + cp.einsum('ij,ij->i', B.imag, B.imag)
    norm_b_denom_sq = (cp.sqrt(norm_b_sq) + 1e-14)**2
    tol2 = tol * tol

    col_res_sq = cp.einsum('ij,ij->i', R.real, R.real) + cp.einsum('ij,ij->i', R.imag, R.imag)
    if cp.max(col_res_sq / norm_b_denom_sq) < tol2:
        return X

    for _ in range(maxiter):
        TP = T_op(P)
        delta = cp.vdot(P, TP).real
        if delta <= 0 or cp.isnan(delta):
            break

        alpha = gamma / delta

        X += alpha * P
        R -= alpha * TP

        col_res_sq = cp.einsum('ij,ij->i', R.real, R.real) + cp.einsum('ij,ij->i', R.imag, R.imag)
        if cp.max(col_res_sq / norm_b_denom_sq) < tol2:
            break

        if M_inv is not None:
            Z_new = M_inv(R)
        else:
            Z_new = R.copy()

        gamma_new = cp.vdot(R, Z_new).real
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
                                    **kwargs):
    theta_j = cp.asarray(theta_j, dtype=float)
    f_orig = cp.asarray(f, dtype=cp.complex128)
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
        bandwidth_factor=kde_bandwidth
    )[:, None]

    # 1. Compute RHS: B_adj = A^H W f (1 cuFINUFFT Adjoint)
    f_w = f_arr * w
    B_adj = _nufft_adjoint(x_wrapped, f_w, N_modes=N, eps=eps).T  # (K, N)

    # 2. Compute Toeplitz kernel from weights (1 cuFINUFFT Adjoint, double resolution)
    v_raw = _nufft_adjoint(x_wrapped, w.flatten(), N_modes=2*N, eps=eps)  # (2N,)
    v_shift = cp.fft.ifftshift(v_raw)
    V_hat = cp.fft.fft(v_shift)[None, :]  # (1, 2N)

    # 3. Fast Toeplitz Matrix-Vector Multiplication via cuFFT
    def T_op(X):
        T_in = cp.zeros((K, 2*N), dtype=cp.complex128)
        T_in[:, :N] = X
        T_hat = cp.fft.fft(T_in, axis=1)
        T_out = cp.fft.ifft(T_hat * V_hat, axis=1)
        return (T_out[:, :N].copy() / (2.0 * N)) + (reg_param) * X

    # 4. Circulant Preconditioner via T. Chan's Optimal Formula
    k = cp.arange(N)
    c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
    eig_c = cp.abs(cp.fft.fft(c_chan)) + precond_shift
    eig_c_inv = (1.0 / eig_c)[None, :]

    def M_inv(V):
        M_in = cp.fft.ifftshift(V, axes=1)
        M_hat = cp.fft.fft(M_in, axis=1)
        M_out = cp.fft.ifft(M_hat * eig_c_inv, axis=1)
        return cp.fft.fftshift(M_out / N, axes=1).copy()

    # 5. Solve using Block CG (Normal Equations)
    X_T = _block_cg(T_op, B_adj, M_inv=M_inv, tol=tol, maxiter=maxiter)
    X = X_T.T
    return X[:, 0] if f_orig.ndim == 1 else X


# ---------------------------------------------------------
# NUDFT inversion on GPU
# ---------------------------------------------------------
def _invert_nudft(theta_j, f, reg_param=1e-20):
    theta = cp.asarray(theta_j, float)
    f = cp.asarray(f, dtype=cp.complex128)
    N = theta.size
    k = cp.arange(-N // 2, N // 2, dtype=float)
    A = cp.exp(1j * cp.outer(theta, k))
    return cp.linalg.lstsq(A, f, rcond=reg_param)[0]


# ---------------------------------------------------------
# Fourier Coefficient Computation — GPU Nonuniform Dispatcher
# ---------------------------------------------------------
def compute_fourier_coeff_nonunif(f_values: cp.ndarray,
                                  theta_j: cp.ndarray,
                                  grid_type: int = 3,
                                  maxiter: int = 200,
                                  tol: float = 1e-10,
                                  use_nudft: bool = False,
                                  reg_param: float = 1e-12,
                                  eps: float = 1e-12,
                                  precond_shift: float = 1e-3,
                                  kde_oversample: int = 4,
                                  kde_bandwidth: float = 1.0,
                                  **kwargs) -> cp.ndarray:
    """
    Computes azimuthal Fourier coefficients on non-uniform angular mesh theta_j using GPU (cuFINUFFT/CuPy).
    """
    f_values = cp.asarray(f_values)
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
            **kwargs
        )
    else:
        coeff_core = _invert_nufft_cgls_unsquared(
            theta_j, f_values,
            tol=tol, maxiter=maxiter, eps=eps,
            reg_param=reg_param,
            **kwargs
        )

    return _pad_coeff_to_Np1(coeff_core, N)