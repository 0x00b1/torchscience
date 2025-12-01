import numpy as np
import torch
from torch import Tensor


def hermite_polynomial_h_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Gauss-Hermite quadrature nodes.

    These are the roots of H_n(x), optimal for polynomial integration
    with respect to the weight function w(x) = exp(-x^2).

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
        Gauss-Hermite nodes, sorted in descending order.

    Notes
    -----
    These points are the roots of the Hermite polynomial H_n(x).
    Together with corresponding weights, they provide exact integration
    for polynomials up to degree 2n-1 with weight w(x) = exp(-x^2).

    The points are symmetric about x=0.

    Examples
    --------
    >>> hermite_polynomial_h_points(3)
    tensor([ 1.2247,  0.0000, -1.2247])
    """
    # Use numpy for high precision computation of roots
    points, _ = np.polynomial.hermite.hermgauss(n)
    # Return in descending order (matching other polynomial conventions)
    return torch.tensor(points[::-1].copy(), dtype=dtype, device=device)
