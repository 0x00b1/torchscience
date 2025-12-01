from typing import Callable

import torch

from ._hermite_polynomial_h import HermitePolynomialH
from ._hermite_polynomial_h_fit import hermite_polynomial_h_fit
from ._hermite_polynomial_h_points import hermite_polynomial_h_points


def hermite_polynomial_h_interpolate(
    f: Callable,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> HermitePolynomialH:
    """Interpolate function at Gauss-Hermite nodes.

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
    HermitePolynomialH
        Interpolating Hermite series.

    Notes
    -----
    Uses Gauss-Hermite nodes (roots of H_n) for optimal conditioning.
    The interpolating polynomial has degree n-1 and passes exactly
    through all n nodes.

    Examples
    --------
    >>> c = hermite_polynomial_h_interpolate(lambda x: x**2, n=3)
    """
    # Generate Gauss-Hermite nodes
    x = hermite_polynomial_h_points(n, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return hermite_polynomial_h_fit(x, y, degree=n - 1)
