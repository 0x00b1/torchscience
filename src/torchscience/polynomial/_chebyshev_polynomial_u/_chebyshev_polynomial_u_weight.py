import warnings

import torch
from torch import Tensor

from ._chebyshev_polynomial_u import ChebyshevPolynomialU


def chebyshev_polynomial_u_weight(
    x: Tensor,
) -> Tensor:
    """Compute Chebyshev U weight function.

    The weight function is w(x) = sqrt(1-x^2), which appears in
    the orthogonality relation for Chebyshev polynomials of the second kind:

        integral_{-1}^{1} U_m(x) U_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points in [-1, 1].

    Returns
    -------
    Tensor
        Weight values w(x) = sqrt(1-x^2).

    Notes
    -----
    The weight is zero at x = +/- 1.

    Examples
    --------
    >>> chebyshev_polynomial_u_weight(torch.tensor([0.0]))
    tensor([1.])
    """
    domain = ChebyshevPolynomialU.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating ChebyshevPolynomialU weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}]. Results may be NaN or complex.",
            stacklevel=2,
        )

    return torch.sqrt(1.0 - x**2)
