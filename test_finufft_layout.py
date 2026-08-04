import numpy as np
import finufft
import time

N = 512
K = 512

x = np.random.uniform(-np.pi, np.pi, N).astype(np.float64)

# C-contiguous
c_c = np.random.randn(K, N) + 1j * np.random.randn(K, N)
# F-contiguous
c_f = np.asfortranarray(c_c)

plan = finufft.Plan(2, (N,), n_trans=K, eps=1e-9, isign=1, dtype='complex128')
plan.setpts(x)

out_c = np.empty((K, N), dtype=np.complex128, order='C')
out_f = np.empty((K, N), dtype=np.complex128, order='F')

t0 = time.time()
plan.execute(c_c, out=out_c)
print("C-contiguous time:", time.time() - t0)

t0 = time.time()
plan.execute(c_f, out=out_f)
print("F-contiguous time:", time.time() - t0)
