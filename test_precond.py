import numpy as np
import scipy.linalg
from Poisson_Solver.grids import generate_jittered_azimuthal, generate_sine_perturbed_azimuthal, generate_clustered_azimuthal
from Poisson_Solver.cpu_solver.fourier.nonuniform import _make_nufft_plans, _block_cgls, _get_density_weights
from Poisson_Solver.cpu_solver.fourier.nonuniform import _make_nufft_plans, _block_cgls
import time

def test_preconditioners(grid_type, N, K):
    print(f"\n--- Testing {grid_type.upper()} Grid (N={N}, K={K}) ---", flush=True)
    
    if grid_type == "jittered":
        theta_j = generate_jittered_azimuthal(N, jitter_fraction=0.45)
    elif grid_type == "sine":
        theta_j = generate_sine_perturbed_azimuthal(N, amplitude=0.4)
    elif grid_type == "clustered":
        theta_j = generate_clustered_azimuthal(N, cluster_strength=1)
    
    theta_j = np.asarray(theta_j, dtype=float)
    x_wrapped = (theta_j + np.pi) % (2 * np.pi) - np.pi
    N_pts = N
    
    np.random.seed(42)
    true_f_arr = np.exp(-((theta_j - np.pi)**2)) * np.cos(3 * theta_j)
    f_arr = np.tile(true_f_arr, (K, 1)).T
    
    w = _get_density_weights(theta_j)[:, None]
    w_row = w.T
    
    plan_fwd, plan_adj = _make_nufft_plans(x_wrapped, N_modes=N, K=K, eps=1e-9)
    
    fwd_out_buf = np.empty((K, N_pts), dtype=np.complex128)
    adj_in_buf = np.empty((K, N_pts), dtype=np.complex128)
    adj_out_buf = np.empty((K, N), dtype=np.complex128)
    
    def A_op(C_block):
        plan_fwd.execute(C_block, out=fwd_out_buf)
        return fwd_out_buf

    def AH_op(D_block):
        np.multiply(D_block, w_row, out=adj_in_buf)
        plan_adj.execute(adj_in_buf, out=adj_out_buf)
        return adj_out_buf

    # 1. Circulant Preconditioner (Strang / PSF)
    ones_vec = np.ones((1, N_pts), dtype=np.complex128)
    c_psf = AH_op(ones_vec)[0, :]
    c_psf_fft = np.fft.ifftshift(c_psf)
    eig_c = np.abs(np.fft.fft(c_psf_fft)) + 1e-3
    eig_c_inv = (1.0 / eig_c)[None, :]
    def M_inv_circ(V):
        V_shift = np.fft.ifftshift(V, axes=1)
        V_hat = np.fft.fft(V_shift, axis=1)
        V_hat *= eig_c_inv
        return np.fft.fftshift(np.fft.ifft(V_hat, axis=1), axes=1)

    # 1b. T. Chan Preconditioner (Bartlett Windowed PSF)
    k_idx = np.arange(-N//2, N//2)
    bartlett_window = 1.0 - np.abs(k_idx) / (N / 2.0)
    c_psf_tchan = c_psf * bartlett_window
    c_psf_tchan_fft = np.fft.ifftshift(c_psf_tchan)
    eig_tchan = np.abs(np.fft.fft(c_psf_tchan_fft)) + 1e-3
    eig_tchan_inv = (1.0 / eig_tchan)[None, :]
    def M_inv_tchan(V):
        V_shift = np.fft.ifftshift(V, axes=1)
        V_hat = np.fft.fft(V_shift, axis=1)
        V_hat *= eig_tchan_inv
        return np.fft.fftshift(np.fft.ifft(V_hat, axis=1), axes=1)

    # 2. None / Jacobi
    M_inv_none = None

    # 3. Banded Cholesky
    b = 5
    ab = np.zeros((b + 1, N), dtype=np.complex128)
    for p in range(b + 1):
        m_p = np.sum(w_row * np.exp(1j * p * theta_j))
        ab[b - p, p:] = m_p
    try:
        c_and_lower = scipy.linalg.cholesky_banded(ab, lower=False)
        def M_inv_banded(V):
            Z_T = scipy.linalg.cho_solve_banded((c_and_lower, False), V.T)
            return Z_T.T
    except np.linalg.LinAlgError:
        print("Banded Cholesky failed")
        M_inv_banded = None

    idx = np.round(x_wrapped * N / (2 * np.pi)).astype(int) % N
    f_weighted = f_arr.T * w_row
    f_unif = np.zeros((K, N), dtype=np.complex128)
    np.add.at(f_unif, (slice(None), idx), f_weighted)
    X_init = np.fft.fftshift(np.fft.fft(f_unif, axis=1), axes=1)

    for name, M_inv in [("Circulant (Strang)", M_inv_circ), ("T. Chan (Bartlett)", M_inv_tchan), ("None / Jacobi", M_inv_none), ("Banded (b=5)", M_inv_banded)]:
        if M_inv is None and name == "Banded (b=5)": continue
        t0 = time.time()
        X = _block_cgls(A_op, AH_op, f_arr.T, M_inv=M_inv, X_init=X_init.copy(), tol=1e-8, maxiter=500, damp=1e-12)
        t1 = time.time()
        if name == "Circulant (Strang)":
            X_ref = X
            err = 0.0
        else:
            err = np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref)
        print(f"{name:20s} | Time: {(t1-t0)*1000:5.1f} ms | RelDiff vs Circulant: {err:.2e}", flush=True)

test_preconditioners("jittered", 128, 128)
test_preconditioners("sine", 128, 128)
test_preconditioners("clustered", 128, 128)
