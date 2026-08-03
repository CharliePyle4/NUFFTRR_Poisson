import numpy as np
import time

N, K = 512, 512
np.random.seed(0)
R = np.random.randn(N, K) + 1j * np.random.randn(N, K)

start = time.time()
for _ in range(1000):
    col_residuals1 = np.sqrt(np.sum(np.abs(R)**2, axis=0))
time1 = time.time() - start

start = time.time()
for _ in range(1000):
    col_residuals2 = np.linalg.norm(R, axis=0)
time2 = time.time() - start

print(f"NumPy naive norm: {time1:.5f}s, sum={np.sum(col_residuals1)}")
print(f"NumPy linalg norm: {time2:.5f}s, sum={np.sum(col_residuals2)}")
print(f"Speedup: {time1/time2:.2f}x")
