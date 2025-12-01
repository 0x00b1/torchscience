from typing import Callable

import torch

from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_fit import jacobi_polynomial_p_fit
from ._jacobi_polynomial_p_points import jacobi_polynomial_p_points


def jacobi_polynomial_p_interpolate(
    f: Callable,
    n: int,
    alpha: float,
    beta: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> JacobiPolynomialP:
    """Interpolate function at Gauss-Jacobi nodes.

    Parameters
    ----------
    f : Callable
        Function to interpolate, f: Tensor -> Tensor.
    n : int
        Number of interpolation points (degree = n-1).
    alpha : float
        Jacobi parameter α, must be > -1.
    beta : float
        Jacobi parameter β, must be > -1.
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    JacobiPolynomialP
        Interpolating Jacobi series.

    Notes
    -----
    Uses Gauss-Jacobi nodes (roots of P_n^{(α,β)}) for optimal conditioning.
    The interpolating polynomial has degree n-1 and passes exactly
    through all n nodes.

    Examples
    --------
    >>> c = jacobi_polynomial_p_interpolate(lambda x: x**2, n=3, alpha=0.0, beta=0.0)
    """
    # Generate Gauss-Jacobi nodes
    x = jacobi_polynomial_p_points(n, alpha, beta, dtype=dtype, device=device)

    # Evaluate function at nodes
    y = f(x)

    # Fit polynomial of degree n-1 (exact interpolation)
    return jacobi_polynomial_p_fit(x, y, degree=n - 1, alpha=alpha, beta=beta)
