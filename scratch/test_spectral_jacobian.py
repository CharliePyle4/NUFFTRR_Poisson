import numpy as np

def compute_spectral_jacobian_weights(theta: np.ndarray) -> np.ndarray:
    """
    Computes spectral Jacobian weights w_j = (2pi/N) * theta'(s_j) for ANY smooth
    non-uniform grid theta_j using 1D FFT spectral differentiation.
    """
    N = theta.size
    s = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # theta(s) - s is strictly 2pi-periodic
    displacement = theta - s
    # Wrap displacement to [-pi, pi) to remove 2pi phase jumps
    displacement = (displacement + np.pi) % (2 * np.pi) - np.pi
    
    # 1D FFT spectral derivative
    k_modes = np.fft.fftfreq(N, 1.0 / N)
    disp_hat = np.fft.fft(displacement)
    disp_prime = np.fft.ifft(1j * k_modes * disp_hat).real
    
    theta_prime = 1.0 + disp_prime
    # Ensure positive weights
    theta_prime = np.maximum(theta_prime, 1e-12)
    w = (2.0 * np.pi / N) * theta_prime
    # Normalize sum to 1.0
    w /= np.sum(w)
    return w

# Test on smooth sine clustered grid: theta(s) = s - alpha * sin(s - theta_0)
N = 64
alpha = 0.40
theta_0 = np.pi
s = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta_exact = s - alpha * np.sin(s - theta_0)
theta_exact = np.mod(theta_exact, 2 * np.pi)
theta_sorted = np.sort(theta_exact)

w_spec = compute_spectral_jacobian_weights(theta_sorted)

# Exact analytical derivative
s_sorted = s # for this monotonic mapping
w_true = (2.0 * np.pi / N) * (1.0 - alpha * np.cos(s - theta_0))
w_true /= np.sum(w_true)

error = np.max(np.abs(w_spec - w_true))
print(f"Max error between 1D FFT spectral derivative weights and analytical Jacobian: {error:.3e}")
