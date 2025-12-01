import math

import torch
from torch import Tensor


def chebyshev_polynomial_v_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Chebyshev V nodes (roots of V_n).

    These are the roots of V_n(x), which are optimal for polynomial
    interpolation with Chebyshev V polynomials.

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
        Chebyshev V nodes.

    Notes
    -----
    The roots of V_n(x) are:
        x_k = cos((2k + 1) * pi / (2n + 1))  for k = 0, 1, ..., n-1

    These are shifted relative to Chebyshev T roots and provide
    good interpolation properties for functions on [-1, 1].

    Examples
    --------
    >>> chebyshev_polynomial_v_points(3)
    """
    k = torch.arange(n, dtype=dtype, device=device)
    x = torch.cos((2 * k + 1) * math.pi / (2 * n + 1))
    return x
