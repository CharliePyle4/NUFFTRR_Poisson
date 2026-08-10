import numpy as np
import scipy.linalg as la
from Poisson_Solver.cpu_solver.fourier.nonuniform import _nufft_adjoint, _compute_fft_kde_weights, _wrap_angles
from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal

N = 128
for kind in ["warped", "multipole", "chebyshev"]:
    theta = generate_fixed_nonuniform_azimuthal(N, kind=kind)
    w = _compute_fft_kde_weights(theta)
    x_wrapped = _wrap_angles(theta)

    k = np.arange(-N//2, N//2)
    A = np.exp(1j * np.outer(theta, k))
    T_exact = A.conj().T @ np.diag(w) @ A

    v_raw = _nufft_adjoint(x_wrapped, w, N_modes=2*N, eps=1e-15)
    
    # 1. Old (uncorrected)
    v_shift_old = np.fft.ifftshift(v_raw)
    V_hat_old = np.fft.fft(v_shift_old)
    
    # 2. New (zeroing out the non-existent lag at index N)
    v_shift_new = v_shift_old.copy()
    v_shift_new[N] = 0.0
    V_hat_new = np.fft.fft(v_shift_new)

    # Test on a vector X:
    X = np.random.randn(N) + 1j * np.random.randn(N)
    T_exact_X = T_exact @ X

    T_in = np.zeros(2*N, dtype=complex)
    T_in[:N] = X
    
    T_old_X = np.fft.ifft(np.fft.fft(T_in) * V_hat_old)[:N]
    T_new_X = np.fft.ifft(np.fft.fft(T_in) * V_hat_new)[:N]

    err_old = np.linalg.norm(T_old_X - T_exact_X) / np.linalg.norm(T_exact_X)
    err_new = np.linalg.norm(T_new_X - T_exact_X) / np.linalg.norm(T_exact_X)
    print(f"Grid: {kind:10s} | T_op error OLD: {err_old:.3e} | T_op error NEW: {err_new:.3e}")
