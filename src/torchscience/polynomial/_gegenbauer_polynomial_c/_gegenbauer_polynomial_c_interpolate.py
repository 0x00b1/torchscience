from typing import Callable

import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
from ._gegenbauer_polynomial_c_fit import gegenbauer_polynomial_c_fit
from ._gegenbauer_polynomial_c_points import gegenbauer_polynomial_c_points


def gegenbauer_polynomial_c_interpolate(
    f: Callable,
    n: int,
    lambda_: Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> GegenbauerPolynomialC:
    """Interpolate function at Gauss-Gegenbauer nodes.

    Parameters
    ----------
    f : Callable
        Function to interpolate, f: Tensor -> Tensor.
    n : int
        Number of interpolation points (degree = n-1).
    lambda_ : Tensor
        Parameter lambda > -1/2.
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    GegenbauerPolynomialC
        Interpolating Gegenbauer series.

    Notes
    -----
    Uses Gauss-Gegenbauer nodes (roots of C_n^{lambda}) for optimal conditioning.
    The interpolating polynomial has degree n-1 and passes exactly
    through all n nodes.

    Examples
    --------
    >>> c = gegenbauer_polynomial_c_interpolate(
    ...     lambda x: x**2, n=3, lambda_=torch.tensor(1.0)
    ... )
    """
    # Ensure lambda_ is a tensor
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(lambda_, dtype=dtype, device=device)

    # Generate Gauss-Gegenbauer nodes
    x = gegenbauer_polynomial_c_points(n, lambda_, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return gegenbauer_polynomial_c_fit(x, y, degree=n - 1, lambda_=lambda_)
