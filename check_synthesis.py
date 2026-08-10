import numpy as np
import finufft
from Poisson_Solver.cpu_solver.fourier.fourier import synthesize_spatial_from_fourier
from Poisson_Solver.cpu_solver.fourier.nonuniform import _pad_coeff_to_Np1, _wrap_angles
from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal

N = 128
theta = generate_fixed_nonuniform_azimuthal(N, kind="warped")
x = _wrap_angles(theta)

k = np.arange(-N//2, N//2)
# Create exact Fourier modes
c_core = np.random.randn(N) + 1j * np.random.randn(N)
c_padded = _pad_coeff_to_Np1(c_core, N)[:, None]

# 1. Current synthesis
u_synth_old = synthesize_spatial_from_fourier(c_padded, theta, N, azu_unif=1, eps=1e-15)[:, 0]

# 2. Corrected synthesis (restoring full Nyquist mode)
c_corrected = c_padded[:N, :].copy()
c_corrected[0, :] += c_padded[N, :] # recombine the two halves of the Nyquist mode!
u_synth_new = finufft.nufft1d2(x, c_corrected.T, isign=+1, eps=1e-15).T[:, 0]

# Exact spatial values
A = np.exp(1j * np.outer(theta, k))
u_exact = A @ c_core

print("Error in OLD synthesis:", np.linalg.norm(u_synth_old - u_exact) / np.linalg.norm(u_exact))
print("Error in NEW synthesis:", np.linalg.norm(u_synth_new - u_exact) / np.linalg.norm(u_exact))
