import sys
sys.path.insert(0, '.')
import time
import numpy as np
from scipy.linalg import lstsq
from Poisson_Solver.grids import generate_jittered_azimuthal

M = 32
print("Testing lstsq flags: Default vs (overwrite_a=True, check_finite=False)")
for N in [64, 128, 256, 512]:
    theta = generate_jittered_azimuthal(N, jitter_fraction=0.25)
    k = np.arange(-N // 2, N // 2, dtype=float)
    A = np.exp(1j * np.outer(theta, k))
    f = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    # 1. Default lstsq
    times_def = []
    for _ in range(5):
        A_copy = A.copy()
        t0 = time.perf_counter()
        c1 = lstsq(A_copy, f, cond=1e-10)[0]
        times_def.append(time.perf_counter() - t0)

    # 2. Optimized flags: overwrite_a=True, check_finite=False
    times_opt = []
    for _ in range(5):
        A_copy = A.copy()
        t0 = time.perf_counter()
        c2 = lstsq(A_copy, f, cond=1e-10, overwrite_a=True, check_finite=False)[0]
        times_opt.append(time.perf_counter() - t0)

    diff = np.max(np.abs(c1 - c2))
    print(f"N={N:3d} | Default: min={min(times_def)*1000:6.2f}ms, std={np.std(times_def)*1000:5.2f}ms | Opt: min={min(times_opt)*1000:6.2f}ms, std={np.std(times_opt)*1000:5.2f}ms | Diff: {diff:.2e}")
