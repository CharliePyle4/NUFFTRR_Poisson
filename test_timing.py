import time
from Poisson_Solver.grids import generate_jittered_azimuthal
from Poisson_Solver.cpu_solver.fourier.nonuniform import _invert_nufft_block_cgls_shared
import numpy as np

N = 32
np.random.seed(0)
theta = generate_jittered_azimuthal(N, 0.35)
f = np.random.randn(N) + 1j * np.random.randn(N)

start = time.time()
c_nufft = _invert_nufft_block_cgls_shared(theta, f, maxiter=200)
print("Time taken:", time.time() - start)

N = 512
np.random.seed(0)
theta = generate_jittered_azimuthal(N, 0.35)
f = np.random.randn(N) + 1j * np.random.randn(N)

start = time.time()
c_nufft = _invert_nufft_block_cgls_shared(theta, f, maxiter=200)
print("Time taken (N=512):", time.time() - start)
