import cupy as cp

from .uniform import compute_C_D_uniform
from .nonuniform import compute_C_D_nonuniform

# CUDA Kernel for Trapezoidal Rule Recurrence
trapezoidal_kernel_code = r'''
extern "C" __global__
void trapezoidal_recurrence(
    double2* v_neg,
    double2* v_pos,
    const double2* C,
    const double2* D,
    const double* r_m,
    int N,
    int M)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int halfN = N / 2;

    if (n > halfN) return;

    // v_neg recurrence (forward)
    double exp_neg = (double)(n - halfN);
    if (M > 1) {
        v_neg[n * M + 1].x = C[n * (M - 1) + 0].x;
        v_neg[n * M + 1].y = C[n * (M - 1) + 0].y;
    }
    for (int i = 2; i < M; ++i) {
        double factor = pow(r_m[i] / r_m[i - 1], exp_neg);
        int current_idx = n * M + i;
        int prev_idx = n * M + i - 1;
        int c_idx = n * (M - 1) + (i - 1);
        v_neg[current_idx].x = factor * v_neg[prev_idx].x + C[c_idx].x;
        v_neg[current_idx].y = factor * v_neg[prev_idx].y + C[c_idx].y;
    }

    // v_pos recurrence (backward)
    double exp_pos = (double)n;
    for (int i = M - 2; i >= 0; --i) {
        double factor = pow(r_m[i] / r_m[i + 1], exp_pos);
        int current_idx = n * M + i;
        int next_idx = n * M + i + 1;
        int d_idx = n * (M - 1) + i;
        v_pos[current_idx].x = factor * v_pos[next_idx].x + D[d_idx].x;
        v_pos[current_idx].y = factor * v_pos[next_idx].y + D[d_idx].y;
    }
}
'''
trapezoidal_kernel = cp.RawKernel(trapezoidal_kernel_code, 'trapezoidal_recurrence')


# CUDA Kernel for Simpson's Rule Recurrence
simpson_kernel_code = r'''
extern "C" __global__
void simpson_recurrence(
    double2* v_neg,
    double2* v_pos,
    const double2* C,
    const double2* D,
    const double* r_m,
    int N,
    int M)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int halfN = N / 2;

    if (n > halfN) return;

    // v_neg recurrence (forward, 2-step)
    double exp_neg = (double)(n - halfN);
    if (M > 1) {
        v_neg[n * M + 1].x = C[n * (M - 1) + 0].x;
        v_neg[n * M + 1].y = C[n * (M - 1) + 0].y;
    }
    for (int i = 2; i < M; ++i) {
        double factor = pow(r_m[i] / r_m[i - 2], exp_neg);
        int current_idx = n * M + i;
        int prev_idx = n * M + i - 2;
        int c_idx = n * (M - 1) + (i - 1);
        v_neg[current_idx].x = factor * v_neg[prev_idx].x + C[c_idx].x;
        v_neg[current_idx].y = factor * v_neg[prev_idx].y + C[c_idx].y;
    }

    // v_pos recurrence (backward, 2-step)
    double exp_pos = (double)n;
    if (M > 1) {
        v_pos[n * M + (M - 2)].x = D[n * (M - 1) + 0].x;
        v_pos[n * M + (M - 2)].y = D[n * (M - 1) + 0].y;
    }
    for (int i = M - 3; i >= 0; --i) {
        double factor = pow(r_m[i] / r_m[i + 2], exp_pos);
        int current_idx = n * M + i;
        int next_idx = n * M + i + 2;
        int d_idx = n * (M - 1) + (i + 1);
        v_pos[current_idx].x = factor * v_pos[next_idx].x + D[d_idx].x;
        v_pos[current_idx].y = factor * v_pos[next_idx].y + D[d_idx].y;
    }
}
'''
simpson_kernel = cp.RawKernel(simpson_kernel_code, 'simpson_recurrence')

def compute_radial_integrals(r_m: cp.ndarray,
                             f_fourier_coeff: cp.ndarray,
                             quad_rule: int,
                             rad_unif: int):
    """
    Dispatch to the appropriate C_n, D_n radial integral routine.

    Parameters
    ----------
    r_m : cp.ndarray, shape (M,)
        Radial grid.
    f_fourier_coeff : cp.ndarray, shape (N+1, M)
        Fourier coefficients f_n(r_m).
    quad_rule : int
        Quadrature rule index (passed through to the underlying routines).
    rad_unif : int
        1 → uniform radial mesh, use compute_C_D_uniform
        0 → nonuniform radial mesh, use compute_C_D_nonuniform

    Returns
    -------
    C, D : cp.ndarray
        Radial integral arrays used in the v^-, v^+ recurrences.
    """
    if rad_unif == 1:
        C, D = compute_C_D_uniform(r_m, f_fourier_coeff, quad_rule)
    elif rad_unif == 0:
        C, D = compute_C_D_nonuniform(r_m, f_fourier_coeff, quad_rule)
    else:
        raise ValueError('Incorrect index for "rad_unif"')
    return C, D


def compute_v_neg_pos(C: cp.ndarray,
                      D: cp.ndarray,
                      r_m: cp.ndarray,
                      N: int,
                      M: int,
                      quad_rule: int):
    """
    Compute v^- and v^+ via radial recurrences.

    Parameters
    ----------
    C, D : cp.ndarray, shape (N/2+1, M-1) or similar
        Radial integral arrays (as produced by compute_C_D_*).
    r_m : cp.ndarray, shape (M,)
        Radial grid.
    N : int
        Number of angular points.
    M : int
        Number of radial points.
    quad_rule : int
        1 → trapezoidal 1-step recurrences
        2 → Simpson 2-step recurrences

    Returns
    -------
    v_neg, v_pos : cp.ndarray, shape (N/2+1, M)
    """
    halfN = N // 2
    v_neg = cp.zeros((halfN + 1, M), dtype=cp.complex128)
    v_pos = cp.zeros((halfN + 1, M), dtype=cp.complex128)

    # Kernel launch configuration
    threads_per_block = 256
    num_modes = halfN + 1
    grid_size = (num_modes + threads_per_block - 1) // threads_per_block

    if quad_rule == 1:
        # Launch the custom kernel for the trapezoidal rule
        trapezoidal_kernel(
            (grid_size,), (threads_per_block,),
            (v_neg, v_pos, C, D, r_m, N, M)
        )
    elif quad_rule == 2:
        # Launch the custom kernel for Simpson's rule
        simpson_kernel(
            (grid_size,), (threads_per_block,),
            (v_neg, v_pos, C, D, r_m, N, M)
        )
    else:
        raise ValueError('Incorrect quad_rule')

    return v_neg, v_pos


def combine_v_neg_pos_to_v(v_neg: cp.ndarray,
                           v_pos: cp.ndarray,
                           r_m: cp.ndarray,
                           N: int,
                           M: int) -> cp.ndarray:
    """
    Combine v^- and v^+ into full v with Hermitian symmetry.

    Parameters
    ----------
    v_neg, v_pos : ndarray, shape (N/2+1, M)
        Outputs of compute_v_neg_pos.
    r_m : cp.ndarray, shape (M,)
        Radial grid.
    N, M : int
        Angular and radial counts.

    Returns
    -------
    v : cp.ndarray, shape (N+1, M)
    """
    halfN = N // 2
    v = cp.zeros((N + 1, M), dtype=complex)

    # central mode (k = 0)
    v[halfN, 0] = v_neg[halfN, 0] + v_pos[0, 0]
    if M > 1:
        v[halfN, 1:] = cp.log(r_m[1:]) * v_neg[halfN, 1:] + v_pos[0, 1:]

    # k = 1..N/2-1 blockwise
    k_idx = cp.arange(1, halfN)
    pos_idx = k_idx
    mir_idx = N - k_idx
    pos_from = halfN - k_idx

    v[pos_idx, :] = v_neg[pos_idx, :] + cp.conj(v_pos[pos_from, :])
    v[mir_idx, :] = cp.conj(v[pos_idx, :])

    return v
