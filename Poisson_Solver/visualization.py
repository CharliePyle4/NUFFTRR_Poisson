"""
Basic visualization utility for plotting 2D disk mesh solutions in 3D.

Use this module to visualize computed or analytic solutions as surfaces on
the (x, y) domain of the disk.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_on_disk(x_coord: np.ndarray, y_coord: np.ndarray, u: np.ndarray, title: str = "Solution on Disk") -> None:
    """
    Plot a 2D solution u(x, y) defined on a polar grid mapped to cartesian coordinates as a
    3D surface plot.

    Parameters
    ----------
    x_coord : ndarray of shape (N, M)
        X-coordinates of the mesh grid (typically from polar-to-Cartesian mapping).
    y_coord : ndarray of shape (N, M)
        Y-coordinates of the mesh grid (same shape as x_coord).
    u : ndarray of shape (N, M)
        Values of the function or solution at each grid point.
        Can be complex; only the real part is plotted.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_on_disk(x_coord, y_coord, u)

    Notes
    -----
    - This function is intended for quick 3D visualization of a solution on the disk.
    - If `u` is complex, only the real part is displayed.
    - The color map is set to "cool" by default.
    """
    # Force data to be real for plotting stability
    u = np.real(u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_coord, y_coord, u, cmap="cool")
    ax.set_title(title)
    plt.show()


def plot_on_disk_with_error(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    u_approx: np.ndarray,
    u_true: np.ndarray
) -> None:
    """
    Plot true solution, approximate solution, and pointwise error on the
    given (x_coord, y_coord) grid, without any alignment transforms.
    """
    # Real parts for plotting
    UT = np.real(u_true)
    UA = np.real(u_approx)

    # Pointwise absolute error
    ptwise_error = np.abs(UT - UA)

    # Optional: append first angular row for continuous-looking rim
    Xp = np.vstack([x_coord, x_coord[0, :]])
    Yp = np.vstack([y_coord, y_coord[0, :]])
    UTp = np.vstack([UT, UT[0, :]])
    UAp = np.vstack([UA, UA[0, :]])
    Ep  = np.vstack([ptwise_error, ptwise_error[0, :]])

    # True solution
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_surface(Xp, Yp, UTp, cmap="cool")
    ax1.set_title("True solution")

    # Approximate solution
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(Xp, Yp, UAp, cmap="cool")
    ax2.set_title("Approximate solution")

    # Error
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot_surface(Xp, Yp, Ep, cmap="cool")
    ax3.set_title(r"$|u_{\mathrm{true}} - u_{\mathrm{approx}}|$")

    plt.show()



import numpy as np

def periodic_trapz_1d(values, theta):
    theta = np.asarray(theta, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()

    if theta.shape[0] != values.shape[0]:
        raise ValueError("theta and values must have the same length")

    # If user included both 0 and 2π, drop the duplicate endpoint
    if np.isclose(theta[0], 0.0) and np.isclose(theta[-1], 2*np.pi):
        theta = theta[:-1]
        values = values[:-1]

    dtheta = np.diff(np.r_[theta, theta[0] + 2*np.pi])
    if np.any(dtheta <= 0):
        raise ValueError("theta must be strictly increasing over one period")

    w = 0.5 * (dtheta + np.r_[dtheta[-1], dtheta[:-1]])
    return np.sum(w * values)

def trap_2d_on_disk(f, r_m, theta_j):
    f = np.asarray(f, dtype=float)
    r_m = np.asarray(r_m, dtype=float).ravel()
    theta_arr = np.asarray(theta_j, dtype=float)

    if f.shape[1] != len(r_m):
        raise ValueError("f must have shape (N_theta, N_r)")

    if theta_arr.ndim == 1:
        if f.shape[0] != len(theta_arr):
            raise ValueError("For 1D theta, f.shape[0] must equal len(theta)")
        ang_int = np.array([
            periodic_trapz_1d(f[:, m], theta_arr) for m in range(len(r_m))
        ])
    elif theta_arr.ndim == 2:
        if f.shape != theta_arr.shape:
            raise ValueError("For 2D theta, f and theta_j must have same shape")
        ang_int = np.array([
            periodic_trapz_1d(f[:, m], theta_arr[:, m]) for m in range(len(r_m))
        ])
    else:
        raise ValueError("theta_j must be 1D or 2D")

    return np.trapz(r_m * ang_int, x=r_m)

def compute_error_metrics(u_approx, u_true, r_m, theta_j):
    ptwise_error = np.abs(u_true - u_approx)

    linf = np.max(ptwise_error)
    true_inf = np.max(np.abs(u_true))
    linf_rel = np.nan if true_inf == 0 else linf / true_inf

    err_sq_int = trap_2d_on_disk(ptwise_error**2, r_m, theta_j)
    true_sq_int = trap_2d_on_disk(np.abs(u_true)**2, r_m, theta_j)

    tol = 1e-13
    if err_sq_int < -tol or true_sq_int < -tol:
        raise ValueError(
            f"Negative integral detected: err_sq_int={err_sq_int}, true_sq_int={true_sq_int}"
        )

    err_sq_int = max(err_sq_int, 0.0)
    true_sq_int = max(true_sq_int, 0.0)

    l2 = np.sqrt(err_sq_int)
    l2_rel = np.nan if true_sq_int == 0 else l2 / np.sqrt(true_sq_int)

    return linf, linf_rel, l2, l2_rel