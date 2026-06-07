import cupy as cp
import finufft


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def _wrap_angles(theta: cp.ndarray) -> cp.ndarray:
    """Wrap angles to [-π, π) for FINUFFT."""
    return (theta + cp.pi) % (2 * cp.pi) - cp.pi


def _is_matrix(a: cp.ndarray) -> bool:
    """Check if array is a matrix (2D with multiple columns)."""
    a = cp.asarray(a)
    return a.ndim == 2 and a.shape[1] > 1


#To implement others