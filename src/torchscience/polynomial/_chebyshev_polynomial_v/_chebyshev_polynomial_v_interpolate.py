from typing import Callable

import torch

from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_fit import chebyshev_polynomial_v_fit
from ._chebyshev_polynomial_v_points import chebyshev_polynomial_v_points


def chebyshev_polynomial_v_interpolate(
    f: Callable,
    n: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> ChebyshevPolynomialV:
    """Interpolate function at Chebyshev V nodes.

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
    ChebyshevPolynomialV
        Interpolating Chebyshev V series.

    Notes
    -----
    Uses Chebyshev V nodes for optimal conditioning. The interpolating
    polynomial has degree n-1 and passes exactly through all n nodes.

    Examples
    --------
    >>> c = chebyshev_polynomial_v_interpolate(lambda x: x**2, n=3)
    """
    # Generate Chebyshev V nodes
    x = chebyshev_polynomial_v_points(n, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return chebyshev_polynomial_v_fit(x, y, degree=n - 1)
