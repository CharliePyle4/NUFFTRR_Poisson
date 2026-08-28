import sys
sys.path.insert(0, '.')
import time
import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fftw_fft
import finufft

from Poisson_Solver.grids import generate_jittered_azimuthal
from Poisson_Solver.cpu_solver.fourier.nonuniform import (
    _wrap_angles,
    _compute_fft_kde_weights,
    _nufft_adjoint,
    _compute_pipe_menon_weights,
)
from Poisson_Solver.cpu_solver.fourier.uniform import compute_fourier_coeff_unif

print("=" * 70)
print("BENCHMARKING OPTIMIZATIONS 2, 3, AND 4 (1 PROCESSOR / 1 THREAD)")
print("=" * 70)

# ==============================================================================
# TEST OPTIMIZATION 2: Toeplitz Shift & Memory Allocation Elimination
# ==============================================================================
def toeplitz_baseline(theta_j, f_arr, tol=1e-8, maxiter=50, eps=1e-10, reg_param=1e-10, precond_shift=1e-3):
    n_threads = 1
    theta_j = np.asarray(theta_j, dtype=float)
    N = theta_j.size
    x_wrapped = _wrap_angles(theta_j)
    N_pts, K = f_arr.shape

    w = _compute_fft_kde_weights(theta_j, oversample=4, bandwidth_factor=1.0, num_processors=n_threads)[:, None]
    f_w = f_arr * w
    B_adj = _nufft_adjoint(x_wrapped, f_w, N_modes=N, eps=eps, num_processors=n_threads).T

    v_raw = _nufft_adjoint(x_wrapped, w.flatten(), N_modes=2*N, eps=eps, num_processors=n_threads)
    v_shift = fftw_fft.ifftshift(v_raw)
    V_hat = fftw_fft.fft(v_shift, threads=n_threads, planner_effort='FFTW_ESTIMATE')[None, :]

    T_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_hat = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    fft_T = pyfftw.FFTW(T_in, T_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    T_ifft_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_out = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    ifft_T = pyfftw.FFTW(T_ifft_in, T_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    def T_op(X):
        T_in[:] = 0.0
        T_in[:, :N] = X
        fft_T.execute()
        T_ifft_in[:] = T_hat * V_hat
        ifft_T.execute()
        return (T_out[:, :N].copy() / (2.0 * N)) + (reg_param) * X

    k = np.arange(N)
    c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
    eig_c = np.abs(fftw_fft.fft(c_chan, threads=n_threads, planner_effort='FFTW_ESTIMATE')) + precond_shift
    eig_c_inv = (1.0 / eig_c)[None, :]

    M_in = pyfftw.empty_aligned((K, N), dtype='complex128')
    M_hat = pyfftw.empty_aligned((K, N), dtype='complex128')
    fft_M = pyfftw.FFTW(M_in, M_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    M_ifft_in = pyfftw.empty_aligned((K, N), dtype='complex128')
    M_out = pyfftw.empty_aligned((K, N), dtype='complex128')
    ifft_M = pyfftw.FFTW(M_ifft_in, M_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    def M_inv(V):
        M_in[:] = fftw_fft.ifftshift(V, axes=1)
        fft_M.execute()
        M_ifft_in[:] = M_hat * eig_c_inv
        ifft_M.execute()
        return fftw_fft.fftshift(M_out, axes=1) / N

    from Poisson_Solver.cpu_solver.fourier.nonuniform import _block_cg
    X_T = _block_cg(T_op, B_adj, M_inv=M_inv, tol=tol, maxiter=maxiter)
    return X_T.T


def toeplitz_optimized(theta_j, f_arr, tol=1e-8, maxiter=50, eps=1e-10, reg_param=1e-10, precond_shift=1e-3):
    n_threads = 1
    theta_j = np.asarray(theta_j, dtype=float)
    N = theta_j.size
    x_wrapped = _wrap_angles(theta_j)
    N_pts, K = f_arr.shape

    w = _compute_fft_kde_weights(theta_j, oversample=4, bandwidth_factor=1.0, num_processors=n_threads)[:, None]
    f_w = f_arr * w
    B_adj = _nufft_adjoint(x_wrapped, f_w, N_modes=N, eps=eps, num_processors=n_threads).T

    v_raw = _nufft_adjoint(x_wrapped, w.flatten(), N_modes=2*N, eps=eps, num_processors=n_threads)
    v_shift = fftw_fft.ifftshift(v_raw)
    V_hat = fftw_fft.fft(v_shift, threads=n_threads, planner_effort='FFTW_ESTIMATE')[None, :]

    # Reusable buffers
    T_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_hat = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    fft_T = pyfftw.FFTW(T_in, T_hat, axes=(1,), direction='FFTW_FORWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    T_ifft_in = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    T_out = pyfftw.empty_aligned((K, 2*N), dtype='complex128')
    ifft_T = pyfftw.FFTW(T_ifft_in, T_out, axes=(1,), direction='FFTW_BACKWARD', threads=n_threads, flags=('FFTW_ESTIMATE',))

    scale_2N = 1.0 / (2.0 * N)
    def T_op(X):
        T_in[:, :N] = X
        T_in[:, N:] = 0.0
        fft_T.execute()
        np.multiply(T_hat, V_hat, out=T_ifft_in)
        ifft_T.execute()
        return (T_out[:, :N] * scale_2N) + (reg_param * X)

    # Pre-shift circulant column once so M_inv requires zero in-loop shifts!
    k = np.arange(N)
    c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
    # Shift to uncentered zero-frequency origin
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

    from Poisson_Solver.cpu_solver.fourier.nonuniform import _block_cg
    X_T = _block_cg(T_op, B_adj, M_inv=M_inv, tol=tol, maxiter=maxiter)
    return X_T.T


print("\n--- Testing Optimization 2: NUFFT Toeplitz Solver ---")
for N in [64, 128, 256]:
    M = N
    theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.25)
    f_arr = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    # Warm-up
    _ = toeplitz_baseline(theta_j, f_arr)
    _ = toeplitz_optimized(theta_j, f_arr)

    # Time baseline
    t0 = time.perf_counter()
    c_base = toeplitz_baseline(theta_j, f_arr)
    t_base = time.perf_counter() - t0

    # Time optimized
    t0 = time.perf_counter()
    c_opt = toeplitz_optimized(theta_j, f_arr)
    t_opt = time.perf_counter() - t0

    max_diff = np.max(np.abs(c_base - c_opt))
    rel_diff = max_diff / np.max(np.abs(c_base))
    speedup = t_base / t_opt

    print(f"N={N:3d}, M={M:3d} | Base: {t_base*1000:6.1f}ms | Opt: {t_opt*1000:6.1f}ms | Speedup: {speedup:.2f}x | Max Diff: {max_diff:.2e} | Rel Diff: {rel_diff:.2e}")


# ==============================================================================
# TEST OPTIMIZATION 3: PCGLS Real^2 + Imag^2 (With Pipe-Menon n_iter=2)
# ==============================================================================
def pcgls_baseline(theta_j, f_arr, tol=1e-8, maxiter=50, eps=1e-10):
    n_threads = 1
    theta = np.asarray(theta_j, dtype=float)
    x = _wrap_angles(theta)
    N = theta.size
    N_pts, K = f_arr.shape
    f_T = np.ascontiguousarray(f_arr.T, dtype=np.complex128)
    c_T = np.zeros((K, N), dtype=np.complex128)
    r_T = f_T.copy()

    # ALWAYS Pipe & Menon with n_iter=2
    w = _compute_pipe_menon_weights(theta, n_iter=2, eps=eps, num_processors=n_threads)[None, :]

    plan1 = finufft.Plan(1, (N,), n_trans=K, isign=-1, eps=eps, nthreads=n_threads)
    plan1.setpts(x)
    plan2 = finufft.Plan(2, (N,), n_trans=K, isign=+1, eps=eps, nthreads=n_threads)
    plan2.setpts(x)

    z_T = np.empty((K, N), dtype=np.complex128)
    np.multiply(r_T, w, out=z_T)
    s_T = np.empty((K, N), dtype=np.complex128)
    plan1.execute(z_T, s_T)

    p_T = s_T.copy()
    q_T = np.empty((K, N), dtype=np.complex128)

    gamma = np.sum(np.abs(s_T)**2, axis=1)
    norm_s0 = np.sqrt(gamma) + 1e-14

    for it in range(maxiter):
        plan2.execute(p_T, q_T)
        norm_q_sq = np.sum(np.abs(q_T)**2 * w, axis=1) + 1e-28
        alpha = (gamma / norm_q_sq)[:, None]
        c_T += alpha * p_T
        r_T -= alpha * q_T
        np.multiply(r_T, w, out=z_T)
        plan1.execute(z_T, s_T)
        gamma_new = np.sum(np.abs(s_T)**2, axis=1)

        rel_res = np.max(np.sqrt(gamma_new) / norm_s0)
        if rel_res < tol:
            break

        beta = (gamma_new / (gamma + 1e-28))[:, None]
        p_T = s_T + beta * p_T
        gamma = gamma_new

    return c_T.T


def pcgls_optimized(theta_j, f_arr, tol=1e-8, maxiter=50, eps=1e-10):
    n_threads = 1
    theta = np.asarray(theta_j, dtype=float)
    x = _wrap_angles(theta)
    N = theta.size
    N_pts, K = f_arr.shape
    f_T = np.ascontiguousarray(f_arr.T, dtype=np.complex128)
    c_T = np.zeros((K, N), dtype=np.complex128)
    r_T = f_T.copy()

    # ALWAYS Pipe & Menon with n_iter=2
    w = _compute_pipe_menon_weights(theta, n_iter=2, eps=eps, num_processors=n_threads)[None, :]

    plan1 = finufft.Plan(1, (N,), n_trans=K, isign=-1, eps=eps, nthreads=n_threads)
    plan1.setpts(x)
    plan2 = finufft.Plan(2, (N,), n_trans=K, isign=+1, eps=eps, nthreads=n_threads)
    plan2.setpts(x)

    z_T = np.empty((K, N), dtype=np.complex128)
    np.multiply(r_T, w, out=z_T)
    s_T = np.empty((K, N), dtype=np.complex128)
    plan1.execute(z_T, s_T)

    p_T = s_T.copy()
    q_T = np.empty((K, N), dtype=np.complex128)

    # Sqr magnitude without hypot/sqrt
    gamma = np.sum(s_T.real**2 + s_T.imag**2, axis=1)
    norm_s0 = np.sqrt(gamma) + 1e-14

    for it in range(maxiter):
        plan2.execute(p_T, q_T)
        norm_q_sq = np.sum((q_T.real**2 + q_T.imag**2) * w, axis=1) + 1e-28
        alpha = (gamma / norm_q_sq)[:, None]
        c_T += alpha * p_T
        r_T -= alpha * q_T
        np.multiply(r_T, w, out=z_T)
        plan1.execute(z_T, s_T)
        gamma_new = np.sum(s_T.real**2 + s_T.imag**2, axis=1)

        rel_res = np.max(np.sqrt(gamma_new) / norm_s0)
        if rel_res < tol:
            break

        beta = (gamma_new / (gamma + 1e-28))[:, None]
        p_T = s_T + beta * p_T
        gamma = gamma_new

    return c_T.T


print("\n--- Testing Optimization 3: PCGLS Sqr-Magnitude (With Pipe-Menon n_iter=2) ---")
for N in [64, 128, 256]:
    M = N
    theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.25)
    f_arr = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    # Warm-up
    _ = pcgls_baseline(theta_j, f_arr)
    _ = pcgls_optimized(theta_j, f_arr)

    # Time baseline
    t0 = time.perf_counter()
    c_base = pcgls_baseline(theta_j, f_arr)
    t_base = time.perf_counter() - t0

    # Time optimized
    t0 = time.perf_counter()
    c_opt = pcgls_optimized(theta_j, f_arr)
    t_opt = time.perf_counter() - t0

    max_diff = np.max(np.abs(c_base - c_opt))
    rel_diff = max_diff / np.max(np.abs(c_base))
    speedup = t_base / t_opt

    print(f"N={N:3d}, M={M:3d} | Base: {t_base*1000:6.1f}ms | Opt: {t_opt*1000:6.1f}ms | Speedup: {speedup:.2f}x | Max Diff: {max_diff:.2e} | Rel Diff: {rel_diff:.2e}")


# ==============================================================================
# TEST OPTIMIZATION 4: Fortran-Contiguous Memory Layout for Step 1
# ==============================================================================
print("\n--- Testing Optimization 4: Fortran Layout for FFT along Axis 0 ---")
for N in [256, 512, 1024]:
    M = N
    f_arr = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    g_arr = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))

    # Baseline C-contiguous
    combined_c = np.hstack([f_arr, g_arr])
    # Fortran-contiguous
    combined_f = np.asfortranarray(np.hstack([f_arr, g_arr]))

    # Time C-contiguous
    t0 = time.perf_counter()
    for _ in range(20):
        res_c = compute_fourier_coeff_unif(combined_c, num_processors=1)
    t_c = (time.perf_counter() - t0) / 20.0

    # Time Fortran-contiguous
    t0 = time.perf_counter()
    for _ in range(20):
        res_f = compute_fourier_coeff_unif(combined_f, num_processors=1)
    t_f = (time.perf_counter() - t0) / 20.0

    max_diff = np.max(np.abs(res_c - res_f))
    speedup = t_c / t_f

    print(f"N={N:4d}, M={M:4d} | C-order: {t_c*1000:5.2f}ms | F-order: {t_f*1000:5.2f}ms | Speedup: {speedup:.2f}x | Max Diff: {max_diff:.2e}")
