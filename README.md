# NUFFTRR_Poisson: High-Performance Spectral Poisson Solver on Disk Domains

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NUFFTRR_Poisson** is a fast, accurate, and scalable spectral solver for the 2D Poisson equation on disk domains in polar coordinates:

$$\Delta u = f \quad \text{on } D = \{(r, \theta) : 0 \le r \le R, 0 \le \theta < 2\pi\}$$

subject to either **Dirichlet** ($u(R, \theta) = g(\theta)$) or **Neumann** ($\frac{\partial u}{\partial r}(R, \theta) = g(\theta)$) boundary conditions.

The solver employs an azimuthal Fourier modal decomposition coupled with radial integration ($C_n, D_n$) solved via recursive quadrature. It supports both **CPU** (multithreaded FFTW and FINUFFT) and **GPU** (CuPy and cuFINUFFT) backends on **uniform** and **arbitrarily non-uniform** polar grids.

---

## Key Features

- **Flexible Angular Discretizations**:
  - **Uniform Angular Grids (`grid_type=1`)**: Classical FFT via FFTW (CPU) or cuFFT (GPU).
  - **Non-Uniform Angular Grids (`grid_type=2`)**: Normal-equations Block-CG solver with FFT-accelerated circular Kernel Density Estimation (KDE) and Tony Chan's optimal circulant preconditioner.
  - **Non-Uniform Angular Grids (`grid_type=3`)**: Unsquared Preconditioned Conjugate Gradient for Least Squares (PCGLS) with Pipe & Menon iterative density compensation.
  - **Direct NUDFT (`use_nudft_angular=True`)**: Dense regularized non-uniform discrete Fourier transform reference solve.
- **Flexible Radial Discretizations**:
  - **Uniform Radial Grids (`rad_unif=1`)** and **Non-Uniform Radial Grids (`rad_unif=0`)**.
  - Vectorized 1-step Trapezoidal (`quad_rule=1`) and 2-step 3-point Simpson variant (`quad_rule=2`) radial recurrences.
- **Dual CPU & GPU Backends**:
  - Switch between CPU and GPU seamlessly with `use_gpu=False` / `use_gpu=True`.
- **Processor & Thread Allocation**:
  - Control multi-threading via `num_processors` (defaults to all available CPU cores).
- **Full Parameter Configurability**:
  - Configurable regularization (`reg_param`), solver tolerances (`tol_nufft`), iteration caps (`maxiter_nufft`), FINUFFT kernel tolerances (`eps_finufft`), KDE oversampling (`kde_oversample`), bandwidth scaling (`kde_bandwidth`), and preconditioner shifts (`precond_shift`).

---

## Installation

### 1. Prerequisites
- Python $\ge$ 3.10
- C/C++ build tools (for FFTW and FINUFFT CPU compilation)
- *(Optional)* NVIDIA CUDA Toolkit $\ge$ 11.8/12.0 for GPU acceleration

### 2. Install Core CPU Dependencies
Clone the repository and install dependencies:

```bash
git clone https://github.com/CharliePyle4/NUFFTRR_Poisson.git
cd NUFFTRR_Poisson
pip install -r requirements.txt
```

### 3. Optional GPU Acceleration
To enable GPU acceleration via CuPy and cuFINUFFT:

```bash
# For CUDA 12.x:
pip install cupy-cuda12x cufinufft

# For CUDA 11.x:
pip install cupy-cuda11x cufinufft
```

---

## Quickstart

### Example 1: Solving a Dirichlet Problem on a Uniform Grid

```python
import numpy as np
from Poisson_Solver.grids import (
    generate_uniform_radial,
    generate_uniform_azimuthal,
    generate_cartesian_grid_on_disk,
    compute_zero_mode,
)
from Poisson_Solver.poisson_solver import poisson_solver

# 1. Define problem setup on disk of radius R = 1
R = 1.0
N, M = 64, 64  # N angular points, M radial points
r_m = generate_uniform_radial(M, R)
theta_j = generate_uniform_azimuthal(N)
x, y = generate_cartesian_grid_on_disk(theta_j, r_m)

# 2. Manufactured solution & RHS
u_exact = 3 * np.exp(x + y) * (x - x**2) * (y - y**2) + 5
f_values = 6 * np.exp(x + y) * x * y * (-3 + x + y + x * y)
g_values = u_exact[:, -1]  # Dirichlet boundary values at r = R
u_0 = compute_zero_mode(u_exact, theta_j, azu_unif=2)

# 3. Solve Poisson equation
u_approx = poisson_solver(
    f_values=f_values,
    g_values=g_values,
    u_fourier_0=u_0,
    N=N,
    M=M,
    r_m=r_m,
    theta_j=theta_j,
    R=R,
    quad_rule=1,       # 1: Trapezoidal, 2: Simpson
    BC_choice=1,       # 1: Dirichlet, 2: Neumann
    rad_unif=1,        # 1: Uniform radial grid
    grid_type=1,       # 1: Uniform angular grid (FFT)
    num_processors=4,  # Use 4 CPU threads (None = all cores)
    use_gpu=False      # True for CUDA execution
)

rel_error = np.linalg.norm(u_approx - u_exact) / np.linalg.norm(u_exact)
print(f"Relative L2 Error: {rel_error:.4e}")
```

### Example 2: Solving on a Non-Uniform Angular Grid (NUFFT)

```python
from Poisson_Solver.grids import generate_fixed_nonuniform_azimuthal

# Non-uniform angular grid
theta_nonunif = generate_fixed_nonuniform_azimuthal(N, p=1.2)
x_nu, y_nu = generate_cartesian_grid_on_disk(theta_nonunif, r_m)
f_nu = 6 * np.exp(x_nu + y_nu) * x_nu * y_nu * (-3 + x_nu + y_nu + x_nu * y_nu)
g_nu = 3 * np.exp(x_nu[:, -1] + y_nu[:, -1]) * (x_nu[:, -1] - x_nu[:, -1]**2) * (y_nu[:, -1] - y_nu[:, -1]**2) + 5
u_exact_nu = 3 * np.exp(x_nu + y_nu) * (x_nu - x_nu**2) * (y_nu - y_nu**2) + 5
u_0_nu = compute_zero_mode(u_exact_nu, theta_nonunif, azu_unif=1)

# Solve using Unsquared PCGLS (grid_type=3) or Block-CG (grid_type=2)
u_approx_nu = poisson_solver(
    f_values=f_nu,
    g_values=g_nu,
    u_fourier_0=u_0_nu,
    N=N,
    M=M,
    r_m=r_m,
    theta_j=theta_nonunif,
    R=R,
    quad_rule=1,
    BC_choice=1,
    rad_unif=1,
    grid_type=3,           # 3: Unsquared PCGLS, 2: Block-CG
    maxiter_nufft=200,
    tol_nufft=1e-10,
    reg_param=1e-12,
    precond_shift=1e-3,
    kde_oversample=4,
    kde_bandwidth=1.0,
    num_processors=None    # Auto-detects all available cores
)
```

---

## API Reference

### `poisson_solver(...)`

```python
def poisson_solver(
    f_values, g_values, u_fourier_0,
    N, M, r_m, theta_j, R,
    quad_rule, BC_choice,
    rad_unif, grid_type,
    use_nudft_angular: bool = False,
    maxiter_nufft: int = 50,
    tol_nufft: float = 1e-8,
    reg_param: float = 1e-12,
    eps_finufft: float = 1e-12,
    precond_shift: float = 1e-3,
    kde_oversample: int = 4,
    kde_bandwidth: float = 1.0,
    num_processors: int = None,
    use_gpu: bool = False,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `f_values` | `ndarray` | *Required* | Forcing values $f(r, \theta)$, shape `(N, M)`. |
| `g_values` | `ndarray` | *Required* | Boundary values $g(\theta)$ at $r = R$, shape `(N,)`. |
| `u_fourier_0` | `complex` or `ndarray` | *Required* | Zero-mode azimuthal average (required for Neumann problem; ignored for Dirichlet). |
| `N` | `int` | *Required* | Number of angular grid points. |
| `M` | `int` | *Required* | Number of radial grid points. |
| `r_m` | `ndarray` | *Required* | Radial grid points spanning $[0, R]$, shape `(M,)`. |
| `theta_j` | `ndarray` | *Required* | Angular grid points in $[0, 2\pi)$, shape `(N,)`. |
| `R` | `float` | *Required* | Radius of the disk domain ($r_M = R$). |
| `quad_rule` | `int` | *Required* | Radial quadrature rule: `1` for Trapezoidal, `2` for 3-point Simpson variant. |
| `BC_choice` | `int` | *Required* | Boundary condition: `1` for Dirichlet, `2` for Neumann. |
| `rad_unif` | `int` | *Required* | Radial grid type: `1` for uniform spacing, `0` for non-uniform spacing. |
| `grid_type` | `int` | *Required* | Angular solver strategy: `1` for uniform FFT, `2` for Toeplitz Block-CG, `3` for Unsquared PCGLS. |
| `use_nudft_angular` | `bool` | `False` | When `True` on non-uniform angles, uses direct dense NUDFT solve instead of NUFFT. |
| `maxiter_nufft` | `int` | `50` | Maximum number of conjugate gradient iterations for NUFFT coefficient recovery. |
| `tol_nufft` | `float` | `1e-8` | Convergence tolerance for iterative NUFFT CG solvers. |
| `reg_param` | `float` | `1e-12` | Tikhonov regularization parameter ($\lambda$) for ill-conditioned angular frames. |
| `eps_finufft` | `float` | `1e-12` | Target kernel precision for FINUFFT / cuFINUFFT transforms. |
| `precond_shift` | `float` | `1e-3` | Spectral shift added to Tony Chan circulant preconditioner eigenvalues in Block-CG. |
| `kde_oversample` | `int` | `4` | Oversampling factor on the fine uniform grid for FFT-accelerated circular KDE. |
| `kde_bandwidth` | `float` | `1.0` | Bandwidth multiplier for wrapped Gaussian smoothing kernel ($\sigma = \text{factor} \times \frac{2\pi}{N}$). |
| `num_processors` | `int` | `None` | Number of threads to use in FFTW / FINUFFT thread pools. `None` defaults to all available CPU cores (`os.cpu_count()`). |
| `use_gpu` | `bool` | `False` | If `True`, executes the entire solver on NVIDIA GPU via CuPy and cuFINUFFT. |

---

## Tests Directory

The `Tests/` directory contains comprehensive test suites, benchmarks, convergence studies, and research paper comparisons:

- **`Tests/CPU/`**:
  - `testing_uniform.ipynb`: Convergence analysis on uniform polar meshes across various problem types.
  - `testing_slight_nonuniform.ipynb`: Solver accuracy on perturbed / slightly non-uniform angular grids.
  - `testing_structured_nonuniform.ipynb`: Analysis on structured non-uniform angular meshes.
  - `testing_badly_nonuniform.ipynb`: Stress tests with extreme angular clustering and gap ratios.
  - `radial_testing.ipynb`: Radial discretization tests comparing Trapezoidal and Simpson rules.
  - `neumann_errors.ipynb`: Systematic Neumann boundary condition verification.
  - `testing_helpers.py`: Core testing harness and benchmark runners.
- **`Tests/CPUvsGPU/`**:
  - `gpu_accuracy.ipynb` & `helpers.py`: Direct CPU vs. GPU parity and speedup benchmarks.
- **`Tests/paper/`**:
  - Jupyter notebooks and helper scripts generating comparison tables and figures.
- **`Tests/JCP_Paper_Comparisons/`**:
  - Benchmarks comparing results directly against published literature (*Journal of Computational Physics*).

You can run any of the notebooks using Jupyter:

```bash
jupyter notebook Tests/CPU/testing_uniform.ipynb
```

---

## References

1. Borges, L., & Daripa, P. *A Fast Parallel Algorithm for the Poisson Equation on a Disk*. Journal of Computational Physics.
2. Barnett, A. H., Magland, J. F., & Klinteberg, L. *Parallel Nonuniform Fast Fourier Transforms on the GPU*. SIAM Journal on Scientific Computing.
3. Pipe, J. G., & Menon, P. *Sampling density compensation in MRI: rationale and an iterative numerical solution*. Magnetic Resonance in Medicine.
