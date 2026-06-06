"""
FFTRR Poisson solver on the unit disk.
"""

from .grids import (
    generate_grid_values,
    generate_nonuniform_radial,
    generate_cartesian_grid_on_disk,
    generate_fixed_nonuniform_azimuthal,
    generate_nonuniform_azimuthal,
    generate_uniform_azimuthal,
    generate_uniform_radial,
    compute_zero_mode
)


from .visualization import (
    trap_2d_on_disk,
    compute_error_metrics,
    plot_on_disk,
    plot_on_disk_with_error,
)

from .cpu_solver.poisson_solver import poisson_solver


__version__ = "0.1.0"

__all__ = [
    "poisson_solver",
    # grids / meshes
    "generate_grid_values",
    "generate_nonuniform_radial",
    "generate_cartesian_grid_on_disk",
    "generate_rand_azimuthal",
    # visualization / metrics
    "trap_2d_on_disk",
    "compute_error_metrics",
    "plot_on_disk",
    "plot_on_disk_with_error",
]
