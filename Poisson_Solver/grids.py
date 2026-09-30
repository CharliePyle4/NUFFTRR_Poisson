import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def generate_uniform_azimuthal(N):
    """
    Uniform azimuthal mesh with N equally spaced angles in [0, 2π).
    """
    return np.linspace(0.0, 2 * np.pi, N, endpoint=False)


def generate_nonuniform_azimuthal(N, M, kind="rand", **kwargs):
    """
    Fully nonuniform azimuthal mesh: different angular mesh for each radius.

    Returns
    -------
    thetas : ndarray, shape (N, M)
        Column ell is the angle vector at radius rho_ell.

    kind in {"rand", "jittered", "clustered", "sine"}.
    """
    thetas = np.zeros((N, M), dtype=float)

    for ell in range(M):
        if kind in ("rand", "random"):
            thetas[:, ell] = generate_rand_azimuthal(N)
        elif kind in ("stratified_rand", "stratified_random", "stratified"):
            thetas[:, ell] = generate_stratified_rand_azimuthal(N)
        elif kind == "jittered":
            jf = kwargs.get("jitter_fraction", 0.35)
            thetas[:, ell] = generate_jittered_azimuthal(N, jitter_fraction=jf)
        elif kind == "clustered":
            cs = kwargs.get("cluster_strength", 2.0)
            center = kwargs.get("center", 0.0)
            thetas[:, ell] = generate_clustered_azimuthal(
                N, cluster_strength=cs, center=center
            )
        elif kind == "sine":
            amp = kwargs.get("amplitude", 0.4)
            mode = kwargs.get("mode", 2)
            thetas[:, ell] = generate_sine_perturbed_azimuthal(
                N, amplitude=amp, mode=mode
            )
        elif kind in ("shock_layer", "arctan", "shock"):
            gamma = kwargs.get("gamma", 0.35)
            theta_0 = kwargs.get("theta_0", np.pi)
            thetas[:, ell] = generate_shock_layer_azimuthal(N, gamma=gamma, theta_0=theta_0)
        elif kind in ("multipole", "multi_pole"):
            thetas[:, ell] = generate_multipole_azimuthal(N)
        elif kind in ("fibonacci", "golden_ratio", "quasi_rand"):
            thetas[:, ell] = generate_fibonacci_azimuthal(N)
        elif kind in ("warped", "conformal"):
            thetas[:, ell] = generate_warped_azimuthal(N)
        elif kind in ("chebyshev", "chebyshev_angular"):
            thetas[:, ell] = generate_chebyshev_angular_azimuthal(N)
        else:
            raise ValueError(
                f"Unknown nonuniform kind '{kind}'. "
                "Valid options are {'rand', 'random', 'stratified_rand', 'jittered', 'clustered', 'sine', 'shock_layer', 'multipole', 'fibonacci', 'warped', 'chebyshev'}."
            )
    return thetas


def generate_fixed_nonuniform_azimuthal(N, kind="rand", **kwargs):
    """
    Shared nonuniform azimuthal mesh: same angle vector for every radius.

    Returns
    -------
    theta : ndarray, shape (N,)
    """
    if kind in ("rand", "random"):
        return generate_rand_azimuthal(N)
    elif kind in ("stratified_rand", "stratified_random", "stratified"):
        return generate_stratified_rand_azimuthal(N)
    elif kind == "jittered":
        jf = kwargs.get("jitter_fraction", 0.35)
        return generate_jittered_azimuthal(N, jitter_fraction=jf)
    elif kind == "clustered":
        cs = kwargs.get("cluster_strength", 2.0)
        center = kwargs.get("center", 0.0)
        return generate_clustered_azimuthal(N, cluster_strength=cs, center=center)
    elif kind == "sine":
        amp = kwargs.get("amplitude", 0.4)
        mode = kwargs.get("mode", 2)
        return generate_sine_perturbed_azimuthal(N, amplitude=amp, mode=mode)
    elif kind in ("shock_layer", "arctan", "shock"):
        gamma = kwargs.get("gamma", 0.35)
        theta_0 = kwargs.get("theta_0", np.pi)
        return generate_shock_layer_azimuthal(N, gamma=gamma, theta_0=theta_0)
    elif kind in ("multipole", "multi_pole"):
        return generate_multipole_azimuthal(N)
    elif kind in ("fibonacci", "golden_ratio", "quasi_rand"):
        return generate_fibonacci_azimuthal(N)
    elif kind in ("warped", "conformal"):
        return generate_warped_azimuthal(N)
    elif kind in ("chebyshev", "chebyshev_angular"):
        return generate_chebyshev_angular_azimuthal(N)
    else:
        raise ValueError(
            f"Unknown fixed nonuniform kind '{kind}'. "
            "Valid options are {'rand', 'random', 'stratified_rand', 'jittered', 'clustered', 'sine', 'shock_layer', 'multipole', 'fibonacci', 'warped', 'chebyshev'}."
        )


def generate_rand_azimuthal(N):
    """
    Generate a completely random azimuthal mesh.

    Angles are sampled independently from Uniform(0, 2π) and then sorted
    so they appear in circular order.
    """
    theta = np.random.uniform(0.0, 2 * np.pi, N)
    theta = np.sort(theta)
    return theta


def generate_stratified_rand_azimuthal(N):
    """
    Generate a stratified random azimuthal mesh.

    Places exactly one independent random angle within each of the N uniform
    sectors [2π j / N, 2π (j+1) / N). This guarantees non-uniform random node
    placements while preventing unsampled spatial gaps, ensuring the Fourier
    matrix remains well-conditioned (κ ~ 1.5) for all N.
    """
    j = np.arange(N, dtype=float)
    sector_starts = 2.0 * np.pi * j / N
    sector_widths = 2.0 * np.pi / N
    offsets = np.random.uniform(0.0, sector_widths, size=N)
    theta = sector_starts + offsets
    return np.sort(theta)


def generate_jittered_azimuthal(N, jitter_fraction=0.35):
    """
    Generate a jittered azimuthal mesh.

    Starts from a uniform grid and perturbs each angle within its local sector.

    Parameters
    ----------
    jitter_fraction : float
        Fraction of one angular cell width used for jitter.
        Should usually be in [0, 0.5). Smaller values look more structured.
    """
    if not (0.0 <= jitter_fraction < 0.5):
        raise ValueError("jitter_fraction must satisfy 0 <= jitter_fraction < 0.5")

    factor = 2 * np.pi / N
    j = np.arange(N)
    base = factor * (j + 0.5)
    delta = np.random.uniform(-jitter_fraction, jitter_fraction, N) * factor
    theta = base + delta
    theta = np.mod(theta, 2 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_clustered_azimuthal(N, cluster_strength=2.0, center=0.0):
    """
    Return a set of N angles in [0, 2π) that are nonuniform,
    clustered near 'center' with controllable strength.
    """
    s = np.linspace(0.0, 1.0, N, endpoint=False)
    s_mapped = s ** cluster_strength
    theta = 2 * np.pi * s_mapped + center
    theta = np.mod(theta, 2 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_sine_perturbed_azimuthal(N, amplitude=0.4, mode=2):
    """
    Nonuniform angles via a smooth sine perturbation of uniform spacing.
    """
    j = np.arange(N)
    factor = 2 * np.pi / N
    base = factor * (j + 0.5)
    theta = base + amplitude * np.sin(mode * base)
    theta = np.mod(theta, 2 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_shock_layer_azimuthal(N, gamma=0.85, theta_0=np.pi):
    """
    Physics-adapted shock / boundary layer grid using conformal arctan mapping.
    Concentrates points near theta_0 with layer width controlled by gamma.
    """
    j = np.arange(N)
    xi = 2.0 * np.pi * (j + 0.5) / N
    d_xi = 0.5 * (xi - theta_0)
    theta = theta_0 + 2.0 * np.arctan(gamma * np.tan(d_xi))
    theta = np.mod(theta, 2.0 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_multipole_azimuthal(N, poles=(2, 4), amps=(0.06, 0.03)):
    """
    Compound harmonic multi-pole grid with multiple localized clustering regions.
    """
    j = np.arange(N)
    xi = 2.0 * np.pi * (j + 0.5) / N
    perturb = sum(amp * np.sin(p * xi) for p, amp in zip(poles, amps))
    theta = np.mod(xi + perturb, 2.0 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_fibonacci_azimuthal(N):
    """
    Golden Ratio / Fibonacci low-discrepancy quasi-random angular grid.
    Provably minimizes discrepancy without creating large empty sampling holes.
    """
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.6180339887
    j = np.arange(N)
    theta = 2.0 * np.pi * np.mod(j * phi, 1.0)
    theta = np.sort(theta)
    return theta


def generate_warped_azimuthal(N, eps1=0.08, eps2=-0.04):
    """
    Smoothly warped conformal/harmonic geometry grid for deformed boundaries.
    """
    j = np.arange(N)
    xi = 2.0 * np.pi * (j + 0.5) / N
    theta = xi + eps1 * np.sin(xi) + eps2 * np.cos(2.0 * xi + 0.5)
    theta = np.mod(theta, 2.0 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_chebyshev_angular_azimuthal(N, alpha=0.18):
    """
    Smooth conformal Chebyshev angular grid on periodic circle [0, 2π).
    Clusters points near interfaces (0, π, 2π) while preserving bounded Jacobian.
    """
    j = np.arange(N)
    xi = 2.0 * np.pi * (j + 0.5) / N
    theta = xi - alpha * np.sin(2.0 * xi)
    theta = np.mod(theta, 2.0 * np.pi)
    theta = np.sort(theta)
    return theta


def generate_uniform_radial(M, R):
    iRadius = np.linspace(0.0, R, M)
    return iRadius


def generate_nonuniform_radial(M, R, mapping=None):
    """
    Generate a nonuniform or random radial mesh on [0, R].
    """
    if mapping in (None, "sqrt", "cubic_root", "atan", "squared"):
        r = np.linspace(0.0, R, M)

    if mapping is None or mapping == "sqrt":
        iRadius = np.sqrt(R) * np.sqrt(r)
    elif mapping == "cubic_root":
        iRadius = R ** (2 / 3) * r ** (1 / 3)
    elif mapping == "atan":
        iRadius = R / np.arctan(R) * np.arctan(r)
    elif mapping == "squared":
        iRadius = r ** 2 / R
    elif mapping == "uniform":
        iRadius = np.linspace(0.0, R, M)
    elif mapping == "random":
        iRadius = np.sort(np.random.uniform(0.0, R, M))
    else:
        raise ValueError(
            f"Unknown mapping: '{mapping}'. Valid options are "
            "'sqrt', 'cubic_root', 'atan', 'squared', 'uniform', 'random'."
        )
    return iRadius


def generate_grid_values(f, x_coord, y_coord):
    """
    Evaluate f(x, y) on a grid or vector of points.

    If f supports NumPy arrays, use that directly.
    Otherwise, fall back to np.vectorize.
    """
    x_coord = np.asarray(x_coord)
    y_coord = np.asarray(y_coord)

    if x_coord.shape != y_coord.shape:
        raise ValueError("x_coord and y_coord must have the same shape")

    try:
        vals = f(x_coord, y_coord)
        return np.asarray(vals)
    except Exception:
        vf = np.vectorize(f)
        return vf(x_coord, y_coord)
    

# ---------------------------------------------------------
# Zero Mode Computation
# ---------------------------------------------------------
def compute_zero_mode(u_true: np.ndarray,
                      theta_j: np.ndarray,
                      azu_unif: int,
                      num_processors: int = None,
                      **kwargs) -> np.ndarray:
    """
    Computes the 0-th Fourier mode (azimuthal average) of the true solution u_true.
    Uses exact spectral/NUDFT Fourier analysis to avoid O(Δθ^2) quadrature bias on non-uniform grids.
    """
    from .cpu_solver.fourier.fourier import compute_angular_fourier_coefficients

    M = u_true.shape[1]
    halfN = u_true.shape[0] // 2
    
    if azu_unif == 2:
        # Uniform mesh: arithmetic mean is exact
        u_fourier_0 = u_true.mean(axis=0)
    else:
        # Nonuniform mesh: use spectral NUDFT/NUFFT analysis
        f_fc, _ = compute_angular_fourier_coefficients(
            f_values=u_true,
            g_values=u_true[:, -1],
            theta_j=theta_j,
            grid_type=1 if azu_unif == 2 else 3,
            use_nudft_angular=True,
            num_processors=num_processors,
            **kwargs
        )
        u_fourier_0 = f_fc[halfN, :]
            
    return u_fourier_0


def pol2cart(rho, phi):
    """
    Convert polar coordinates (rho, phi) to Cartesian coordinates (x, y).
    """
    rho = np.asarray(rho)
    phi = np.asarray(phi)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def generate_cartesian_grid_on_disk(iAngle, iRadius):
    """
    Generate Cartesian (x, y) grid coordinates for a disk.

    - If iAngle is 1D (N,), uses the same angles for all radii.
    - If iAngle is 2D (N, M), assumes column ell gives angles for radius iRadius[ell].
    """
    iAngle = np.asarray(iAngle)
    iRadius = np.asarray(iRadius)

    if iAngle.ndim == 1:
        rho, phi = np.meshgrid(iRadius, iAngle, indexing="xy")
    elif iAngle.ndim == 2:
        phi = iAngle
        rho = np.broadcast_to(iRadius[None, :], phi.shape)
    else:
        raise ValueError("iAngle must be 1D or 2D")

    x_coord = rho * np.cos(phi)
    y_coord = rho * np.sin(phi)
    return x_coord, y_coord



def plot_grid(ax, theta, radii, R, title, color):
    """
    Plot the grid using the 'second plot' style:
    - 1D theta (uniform / fixed nonuniform): segmented spokes between radii
    - 2D theta (fully nonuniform): each point gets its own inward segment
      ending at the previous radius
    """
    x, y = generate_cartesian_grid_on_disk(theta, radii)

    # concentric circles
    for r in radii[1:]:
        ax.add_patch(Circle((0, 0), r, fill=False, ec="0.8", lw=0.5))

    if np.ndim(theta) == 1:
        # Uniform / fixed nonuniform:
        # from center to first nonzero radius, then segment between each pair of radii
        for th in theta:
            if len(radii) > 1:
                ax.plot(
                    [0, radii[1] * np.cos(th)],
                    [0, radii[1] * np.sin(th)],
                    color="0.8", lw=0.5
                )

            for k in range(2, len(radii)):
                r_inner = radii[k - 1]
                r_outer = radii[k]
                ax.plot(
                    [r_inner * np.cos(th), r_outer * np.cos(th)],
                    [r_inner * np.sin(th), r_outer * np.sin(th)],
                    color="0.8", lw=0.5
                )

    else:
        # Fully nonuniform:
        # each sample point gets its own inward segment, cut off at previous radius
        Ntheta, Mrad = theta.shape

        for ell in range(1, Mrad):
            r_inner = radii[ell - 1]
            r_outer = radii[ell]

            for j in range(Ntheta):
                th = theta[j, ell]
                ax.plot(
                    [r_inner * np.cos(th), r_outer * np.cos(th)],
                    [r_inner * np.sin(th), r_outer * np.sin(th)],
                    color="0.8", lw=0.5
                )

    # sample points
    ax.scatter(x, y, s=6, color=color, zorder=3)

    # outer boundary
    ax.add_patch(Circle((0, 0), R, fill=False, ec="black", lw=1.0))

    ax.set_title(title)
    ax.set_xlim(-1.05 * R, 1.05 * R)
    ax.set_ylim(-1.05 * R, 1.05 * R)
    ax.set_aspect("equal")
    ax.axis("off")