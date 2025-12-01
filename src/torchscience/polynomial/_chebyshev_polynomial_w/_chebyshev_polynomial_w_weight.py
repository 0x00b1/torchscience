import warnings

import torch
from torch import Tensor

from ._chebyshev_polynomial_w import ChebyshevPolynomialW


def chebyshev_polynomial_w_weight(
    x: Tensor,
) -> Tensor:
    """Compute Chebyshev W weight function.

    The weight function is w(x) = sqrt((1-x)/(1+x)), which appears in
    the orthogonality relation for Chebyshev W polynomials:

        integral_{-1}^{1} W_m(x) W_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points in (-1, 1).

    Returns
    -------
    Tensor
        Weight values w(x) = sqrt((1-x)/(1+x)).

    Notes
    -----
    The weight is undefined at x = ±1. Points near x=-1 will have
    very large weights, and points near x=1 will have weights near 0.

    This is the reciprocal of the Chebyshev V weight function.

    Examples
    --------
    >>> chebyshev_polynomial_w_weight(torch.tensor([0.0]))
    tensor([1.])
    """
    domain = ChebyshevPolynomialW.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating ChebyshevPolynomialW weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}]. Results may be NaN or complex.",
            stacklevel=2,
        )

    return torch.sqrt((1.0 - x) / (1.0 + x))
