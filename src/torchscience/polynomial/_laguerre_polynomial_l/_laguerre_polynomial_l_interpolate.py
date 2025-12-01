from typing import Callable

import torch

from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_fit import laguerre_polynomial_l_fit
from ._laguerre_polynomial_l_points import laguerre_polynomial_l_points


def laguerre_polynomial_l_interpolate(
    f: Callable,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> LaguerrePolynomialL:
    """Interpolate function at Gauss-Laguerre nodes.

    Parameters
    ----------
    f : Callable
        Function to interpolate, f: Tensor -> Tensor.
    n : int
        Number of interpolation points (degree = n-1).
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    LaguerrePolynomialL
        Interpolating Laguerre series.

    Notes
    -----
    Uses Gauss-Laguerre nodes (roots of L_n) for optimal conditioning.
    The interpolating polynomial has degree n-1 and passes exactly
    through all n nodes.

    Examples
    --------
    >>> c = laguerre_polynomial_l_interpolate(lambda x: torch.exp(-x), n=5)
    """
    # Generate Gauss-Laguerre nodes
    x = laguerre_polynomial_l_points(n, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return laguerre_polynomial_l_fit(x, y, degree=n - 1)
