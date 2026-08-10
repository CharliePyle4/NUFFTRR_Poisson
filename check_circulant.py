import numpy as np
import scipy.linalg as la
from Poisson_Solver.cpu_solver.fourier.nonuniform import _nufft_adjoint, _compute_fft_kde_weights, _wrap_angles
from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal

N = 8
theta = generate_fixed_nonuniform_azimuthal(N, kind="warped")
w = _compute_fft_kde_weights(theta)
x_wrapped = _wrap_angles(theta)

k = np.arange(-N//2, N//2)
A = np.exp(1j * np.outer(theta, k))
T_exact = A.conj().T @ np.diag(w) @ A

# Toeplitz entries:
# T_exact[j, k] is Toeplitz because (A^H W A)_{j,k} = sum_m w_m exp(i (k-j) theta_m)
# First col t[0..N-1]:
t_col = T_exact[:, 0]  # k-j = -j, j=0..N-1 -> lag 0, -1, -2, ..., -(N-1)
# First row t[0..N-1]:
t_row = T_exact[0, :]  # k-j = k, k=0..N-1 -> lag 0, 1, 2, ..., N-1

# Standard Circulant embedding of Toeplitz matrix (2N x 2N):
# c = [t_0, t_1, ..., t_{N-1}, 0, t_{-(N-1)}, ..., t_{-1}]
c_circ = np.zeros(2*N, dtype=complex)
c_circ[:N] = t_col
c_circ[N+1:] = t_row[1:][::-1]

# What does current code do?
v_raw = _nufft_adjoint(x_wrapped, w, N_modes=2*N, eps=1e-15)
# v_raw has modes k = -N, ..., N-1
v_shift = np.fft.ifftshift(v_raw)

print("Standard circulant embedding c_circ:")
print(np.round(c_circ, 3))
print("\nCurrent v_shift from ifftshift(v_raw):")
print(np.round(v_shift, 3))
print("\nDifference:", np.linalg.norm(c_circ - v_shift))
