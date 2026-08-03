import numpy as np
import time

N, K = 512, 512
np.random.seed(0)
P = np.random.randn(N, K) + 1j * np.random.randn(N, K)
T_P = np.random.randn(N, K) + 1j * np.random.randn(N, K)

start = time.time()
for _ in range(1000):
    delta1 = np.sum(np.real(np.conj(P) * T_P))
time1 = time.time() - start

start = time.time()
for _ in range(1000):
    delta2 = np.vdot(P, T_P).real
time2 = time.time() - start

print(f"NumPy naive sum: {time1:.5f}s, result: {delta1}")
print(f"NumPy vdot:      {time2:.5f}s, result: {delta2}")
print(f"Speedup: {time1/time2:.2f}x")
