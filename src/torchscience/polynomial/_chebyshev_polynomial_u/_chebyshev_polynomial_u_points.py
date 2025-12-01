import math

import torch
from torch import Tensor


def chebyshev_polynomial_u_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Chebyshev nodes of the second kind.

    These are the roots of U_n(x), optimal for polynomial interpolation
    with Chebyshev polynomials of the second kind.

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
        Chebyshev U nodes x_k = cos(k*pi/(n+1)) for k = 1, 2, ..., n.

    Notes
    -----
    These points are the roots of U_n(x) and are optimal for interpolation
    with Chebyshev U polynomials on [-1, 1].

    The roots of U_n(x) are x_k = cos(k*pi/(n+1)) for k = 1, ..., n.

    Examples
    --------
    >>> chebyshev_polynomial_u_points(3)
    tensor([ 0.7071,  0.0000, -0.7071])
    """
    k = torch.arange(1, n + 1, dtype=dtype, device=device)
    x = torch.cos(k * math.pi / (n + 1))
    return x
