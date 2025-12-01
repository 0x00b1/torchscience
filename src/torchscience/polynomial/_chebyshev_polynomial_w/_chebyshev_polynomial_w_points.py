import math

import torch
from torch import Tensor


def chebyshev_polynomial_w_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Chebyshev W nodes (roots of W_n).

    These are the roots of W_n(x), which are optimal for polynomial
    interpolation with Chebyshev W polynomials.

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
        Chebyshev W nodes.

    Notes
    -----
    The roots of W_n(x) are:
        x_k = cos(2k * pi / (2n + 1))  for k = 1, 2, ..., n

    These are the zeros of the Chebyshev polynomial of the fourth kind
    and provide good interpolation properties for functions on [-1, 1].

    Examples
    --------
    >>> chebyshev_polynomial_w_points(3)
    """
    k = torch.arange(1, n + 1, dtype=dtype, device=device)
    x = torch.cos(2 * k * math.pi / (2 * n + 1))
    return x
