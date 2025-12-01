import warnings

import torch
from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP


def legendre_polynomial_p_weight(
    x: Tensor,
) -> Tensor:
    """Compute Legendre weight function.

    The weight function is w(x) = 1, which appears in the orthogonality
    relation for Legendre polynomials:

        integral_{-1}^{1} P_m(x) P_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate weight.

    Returns
    -------
    Tensor
        Weight values w(x) = 1 (all ones with same shape as x).

    Notes
    -----
    Unlike Chebyshev polynomials which have w(x) = 1/sqrt(1-x^2),
    Legendre polynomials are orthogonal with respect to the uniform
    weight w(x) = 1 on [-1, 1].

    Examples
    --------
    >>> legendre_polynomial_p_weight(torch.tensor([0.0, 0.5, 1.0]))
    tensor([1., 1., 1.])
    """
    domain = LegendrePolynomialP.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating LegendrePolynomialP weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}].",
            stacklevel=2,
        )

    return torch.ones_like(x)
