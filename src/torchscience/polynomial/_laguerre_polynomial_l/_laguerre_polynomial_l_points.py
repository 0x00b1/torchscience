import numpy as np
import torch
from torch import Tensor


def laguerre_polynomial_l_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Gauss-Laguerre quadrature nodes.

    These are the roots of L_n(x), optimal for polynomial integration
    with weight exp(-x) on [0, ∞).

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
        Gauss-Laguerre nodes in [0, ∞), sorted in descending order.

    Notes
    -----
    These points are the roots of the Laguerre polynomial L_n(x).
    Together with corresponding weights, they provide exact integration
    for polynomials up to degree 2n-1 with weight exp(-x).

    All points are positive since L_n(x) has n real positive roots.

    Examples
    --------
    >>> laguerre_polynomial_l_points(3)
    tensor([6.2899, 2.2943, 0.4158])
    """
    # Use numpy for high precision computation of roots
    points, _ = np.polynomial.laguerre.laggauss(n)
    # Return in descending order (matching convention in other polynomial classes)
    return torch.tensor(points[::-1].copy(), dtype=dtype, device=device)
