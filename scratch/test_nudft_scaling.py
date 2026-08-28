import sys
sys.path.insert(0, '.')
import time
import numpy as np
from scipy.linalg import lstsq
from Poisson_Solver.grids import generate_jittered_azimuthal

M = 32
print(f"Benchmarking NUDFT runtime for fixed M = {M} across N:")
for N in [32, 64, 128, 256, 512, 1024]:
    theta = generate_jittered_azimuthal(N, jitter_fraction=0.25)
    k = np.arange(-N // 2, N // 2, dtype=float)
    
    # Measure outer product creation
    t0 = time.perf_counter()
    for _ in range(5):
        A = np.exp(1j * np.outer(theta, k))
    t_matrix = (time.perf_counter() - t0) / 5.0
    
    f = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    
    # Warmup
    _ = lstsq(A, f, cond=1e-10)
    
    # Measure lstsq
    runtimes = []
    for _ in range(5):
        t0 = time.perf_counter()
        c = lstsq(A, f, cond=1e-10)[0]
        t1 = time.perf_counter()
        runtimes.append(t1 - t0)
    
    min_t = min(runtimes)
    mean_t = np.mean(runtimes)
    std_t = np.std(runtimes)
    
    print(f"N={N:4d}, M={M:2d} | Matrix Gen: {t_matrix*1000:6.2f}ms | lstsq Min: {min_t*1000:7.2f}ms | Mean: {mean_t*1000:7.2f}ms | Std: {std_t*1000:5.2f}ms")
