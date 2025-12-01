from typing import Callable

import torch

from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_fit import legendre_polynomial_p_fit
from ._legendre_polynomial_p_points import legendre_polynomial_p_points


def legendre_polynomial_p_interpolate(
    f: Callable,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> LegendrePolynomialP:
    """Interpolate function at Gauss-Legendre nodes.

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
    LegendrePolynomialP
        Interpolating Legendre series.

    Notes
    -----
    Uses Gauss-Legendre nodes (roots of P_n) for optimal conditioning.
    The interpolating polynomial has degree n-1 and passes exactly
    through all n nodes.

    Examples
    --------
    >>> c = legendre_polynomial_p_interpolate(lambda x: x**2, n=3)
    """
    # Generate Gauss-Legendre nodes
    x = legendre_polynomial_p_points(n, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return legendre_polynomial_p_fit(x, y, degree=n - 1)
