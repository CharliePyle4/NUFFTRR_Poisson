import numpy as np

# Test the exact math of original GPU Toeplitz vs modified
N = 32
K = 32
w = np.random.rand(N)
v_raw = np.random.randn(2 * N) + 1j * np.random.randn(2 * N)

# Original GPU M_inv:
k = np.arange(N)
c_chan = ((N - k) / N) * v_raw[N : 2*N] + (k / N) * v_raw[0 : N]
eig_c_orig = np.abs(np.fft.fft(c_chan)) + 1e-3
eig_c_inv_orig = (1.0 / eig_c_orig)[None, :]

V = np.random.randn(K, N) + 1j * np.random.randn(K, N)

# Orig:
M_in_orig = np.fft.ifftshift(V, axes=1)
M_hat_orig = np.fft.fft(M_in_orig, axis=1)
M_out_orig = np.fft.ifft(M_hat_orig * eig_c_inv_orig, axis=1)
res_orig = np.fft.fftshift(M_out_orig, axes=1)

# Modified:
c_chan_shift = np.fft.ifftshift(c_chan)
eig_c_mod = np.abs(np.fft.fft(c_chan_shift)) + 1e-3
eig_c_inv_mod = (1.0 / eig_c_mod)[None, :]
M_hat_mod = np.fft.fft(V, axis=1)
res_mod = np.fft.ifft(M_hat_mod * eig_c_inv_mod, axis=1)

diff = np.max(np.abs(res_orig - res_mod))
print(f"M_inv difference: {diff:.2e}")
