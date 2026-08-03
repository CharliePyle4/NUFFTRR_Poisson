import numpy as np

K = 2
N = 5
N_pts = 4
idx = np.array([0, 1, 0, 4])
f = np.array([[10, 20, 30, 40], [1, 2, 3, 4]])

f_unif = np.zeros((K, N))
np.add.at(f_unif, (slice(None), idx), f)

print(f_unif)
