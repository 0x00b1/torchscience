import numpy as np
import torch
from torch import Tensor


def legendre_polynomial_p_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Gauss-Legendre quadrature nodes.

    These are the roots of P_n(x), optimal for polynomial integration
    and interpolation on [-1, 1].

    Parameters
    ----------
    n : int
        Number of points.
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    Tensor
        Gauss-Legendre nodes in [-1, 1], sorted in descending order.

    Notes
    -----
    These points are the roots of the Legendre polynomial P_n(x).
    Together with corresponding weights, they provide exact integration
    for polynomials up to degree 2n-1.

    The points are symmetric about x=0.

    Examples
    --------
    >>> legendre_polynomial_p_points(3)
    tensor([ 0.7746,  0.0000, -0.7746])
    """
    # Use numpy for high precision computation of roots
    points, _ = np.polynomial.legendre.leggauss(n)
    # Return in descending order (matching Chebyshev convention)
    return torch.tensor(points[::-1].copy(), dtype=dtype, device=device)
