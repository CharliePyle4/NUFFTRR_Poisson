import numpy as np

def compute_spectral_jacobian_weights(theta: np.ndarray) -> np.ndarray:
    N = theta.size
    s = np.linspace(0, 2 * np.pi, N, endpoint=False)
    displacement = (theta - s + np.pi) % (2 * np.pi) - np.pi
    k_modes = np.fft.fftfreq(N, 1.0 / N)
    theta_prime = 1.0 + np.fft.ifft(1j * k_modes * np.fft.fft(displacement)).real
    w = (2.0 * np.pi / N) * np.maximum(theta_prime, 1e-14)
    w /= np.sum(w)
    return w

def compute_geometric_weights(theta: np.ndarray) -> np.ndarray:
    th_sort_idx = np.argsort(theta)
    th_sort = theta[th_sort_idx]
    th_ext = np.concatenate(([th_sort[-1] - 2.0 * np.pi], th_sort, [th_sort[0] + 2.0 * np.pi]))
    w_sort = (th_ext[2:] - th_ext[:-2]) / (4.0 * np.pi)
    w = np.zeros_like(theta)
    w[th_sort_idx] = w_sort
    return w

# Compare on Random Grid
np.random.seed(42)
N = 64
th_rand = np.sort(np.random.uniform(0, 2 * np.pi, N))

w_geom = compute_geometric_weights(th_rand)
w_spec = compute_spectral_jacobian_weights(th_rand)

print("--- Random Grid Weights Comparison ---")
print("Geometric weights: min =", np.min(w_geom), "max =", np.max(w_geom), "std =", np.std(w_geom))
print("Spectral FFT weights: min =", np.min(w_spec), "max =", np.max(w_spec), "std =", np.std(w_spec))

# Compare on Jittered Grid (small perturbation)
th_unif = np.linspace(0, 2 * np.pi, N, endpoint=False)
th_jit = th_unif + 0.1 * (2 * np.pi / N) * np.random.uniform(-1, 1, N)
th_jit = np.sort(np.mod(th_jit, 2 * np.pi))

w_geom_jit = compute_geometric_weights(th_jit)
w_spec_jit = compute_spectral_jacobian_weights(th_jit)

print("\n--- Jittered Grid Weights Comparison ---")
print("Geometric weights: min =", np.min(w_geom_jit), "max =", np.max(w_geom_jit), "std =", np.std(w_geom_jit))
print("Spectral FFT weights: min =", np.min(w_spec_jit), "max =", np.max(w_spec_jit), "std =", np.std(w_spec_jit))
