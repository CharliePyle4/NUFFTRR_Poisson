import sys
sys.path.insert(0, '.')
import numpy as np
from Poisson_Solver.grids import generate_jittered_azimuthal
from Poisson_Solver.cpu_solver.fourier.nonuniform import _wrap_angles, _compute_fft_kde_weights, _nufft_adjoint, _block_cg

N = 64
K = 64
theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.25)
f_arr = np.random.randn(N, K) + 1j * np.random.randn(N, K)
reg_param = 1e-10
precond_shift = 1e-3
tol = 1e-8
maxiter = 50
eps = 1e-10

x_wrapped = _wrap_angles(theta_j)
w = _compute_fft_kde_weights(theta_j, oversample=4, bandwidth_factor=1.0, num_processors=1)[:, None]
f_w = f_arr * w
B_adj = _nufft_adjoint(x_wrapped, f_w, N_modes=N, eps=eps, num_processors=1).T

v_raw = _nufft_adjoint(x_wrapped, w.flatten(), N_modes=2*N, eps=eps, num_processors=1)
v_shift = np.fft.ifftshift(v_raw)
V_hat = np.fft.fft(v_shift)[None, :]

# 1. Original GPU Toeplitz (using NumPy FFT)
def T_op_orig(X):
    T_in = np.zeros((K, 2*N), dtype=np.complex128)
    T_in[:, :N] = X
    T_hat = np.fft.fft(T_in, axis=1)
    T_out = np.fft.ifft(T_hat * V_hat, axis=1)
    return T_out[:, :N].copy() + (reg_param) * X

k = np.arange(N)
c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
eig_c_orig = np.abs(np.fft.fft(c_chan)) + precond_shift
eig_c_inv_orig = (1.0 / eig_c_orig)[None, :]

def M_inv_orig(V):
    M_in = np.fft.ifftshift(V, axes=1)
    M_hat = np.fft.fft(M_in, axis=1)
    M_out = np.fft.ifft(M_hat * eig_c_inv_orig, axis=1)
    return np.fft.fftshift(M_out, axes=1).copy()

X_orig = _block_cg(T_op_orig, B_adj, M_inv=M_inv_orig, tol=tol, maxiter=maxiter)

# 2. Buggy version with extra scale_2N and scale_N
scale_2N = 1.0 / (2.0 * N)
scale_N = 1.0 / N
T_in_bug = np.zeros((K, 2*N), dtype=np.complex128)
def T_op_bug(X):
    T_in_bug[:, :N] = X
    T_in_bug[:, N:] = 0.0
    T_hat = np.fft.fft(T_in_bug, axis=1)
    T_out = np.fft.ifft(T_hat * V_hat, axis=1)
    return (T_out[:, :N] * scale_2N) + (reg_param * X)

c_chan_shift = np.fft.ifftshift(c_chan)
eig_c_shift = np.abs(np.fft.fft(c_chan_shift)) + precond_shift
eig_c_inv_shift = (1.0 / eig_c_shift)[None, :]

def M_inv_bug(V):
    M_hat = np.fft.fft(V, axis=1)
    M_out = np.fft.ifft(M_hat * eig_c_inv_shift, axis=1)
    return M_out * scale_N

X_bug = _block_cg(T_op_bug, B_adj, M_inv=M_inv_bug, tol=tol, maxiter=maxiter)

# 3. Corrected Optimized version (pre-allocated buffer, zero in-loop shifts, NO extra scale)
T_in_correct = np.zeros((K, 2*N), dtype=np.complex128)
def T_op_correct(X):
    T_in_correct[:, :N] = X
    T_in_correct[:, N:] = 0.0
    T_hat = np.fft.fft(T_in_correct, axis=1)
    T_out = np.fft.ifft(T_hat * V_hat, axis=1)
    return T_out[:, :N] + (reg_param * X)

def M_inv_correct(V):
    M_hat = np.fft.fft(V, axis=1)
    M_out = np.fft.ifft(M_hat * eig_c_inv_shift, axis=1)
    return M_out

X_correct = _block_cg(T_op_correct, B_adj, M_inv=M_inv_correct, tol=tol, maxiter=maxiter)

diff_bug = np.max(np.abs(X_orig - X_bug))
diff_correct = np.max(np.abs(X_orig - X_correct))
print(f"Difference (Orig vs Buggy extra scaling):  {diff_bug:.2e}")
print(f"Difference (Orig vs Corrected Optimized):   {diff_correct:.2e}")
