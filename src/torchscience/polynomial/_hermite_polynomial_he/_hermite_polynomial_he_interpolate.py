from typing import Callable

import torch

from ._hermite_polynomial_he import HermitePolynomialHe
from ._hermite_polynomial_he_fit import hermite_polynomial_he_fit
from ._hermite_polynomial_he_points import hermite_polynomial_he_points


def hermite_polynomial_he_interpolate(
    f: Callable,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> HermitePolynomialHe:
    """Interpolate function at Gauss-Hermite_e nodes.

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
    HermitePolynomialHe
        Interpolating Hermite series.

    Notes
    -----
    Uses Gauss-Hermite_e nodes (roots of He_n) for optimal conditioning.
    The interpolating polynomial has degree n-1 and passes exactly
    through all n nodes.

    Examples
    --------
    >>> c = hermite_polynomial_he_interpolate(lambda x: x**2, n=3)
    """
    # Generate Gauss-Hermite_e nodes
    x = hermite_polynomial_he_points(n, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return hermite_polynomial_he_fit(x, y, degree=n - 1)
