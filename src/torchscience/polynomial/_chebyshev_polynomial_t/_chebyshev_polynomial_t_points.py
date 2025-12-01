import math

import torch
from torch import Tensor


def chebyshev_polynomial_t_points(
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Chebyshev nodes of the first kind.

    These are the roots of T_n(x), optimal for polynomial interpolation.

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
        Chebyshev nodes x_k = cos((2k+1)*pi/(2n)) for k = 0, 1, ..., n-1.

    Notes
    -----
    These points minimize the Lebesgue constant for polynomial interpolation
    on [-1, 1] and are the roots of T_n(x).

    Examples
    --------
    >>> chebyshev_polynomial_t_points(3)
    tensor([ 0.8660,  0.0000, -0.8660])
    """
    k = torch.arange(n, dtype=dtype, device=device)
    x = torch.cos((2 * k + 1) * math.pi / (2 * n))
    return x
