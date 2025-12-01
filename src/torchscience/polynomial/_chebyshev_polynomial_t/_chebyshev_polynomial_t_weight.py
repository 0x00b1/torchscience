import warnings

import torch
from torch import Tensor

from ._chebyshev_polynomial_t import ChebyshevPolynomialT


def chebyshev_polynomial_t_weight(
    x: Tensor,
) -> Tensor:
    """Compute Chebyshev weight function.

    The weight function is w(x) = 1/sqrt(1-x^2), which appears in
    the orthogonality relation for Chebyshev polynomials:

        integral_{-1}^{1} T_m(x) T_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points in (-1, 1).

    Returns
    -------
    Tensor
        Weight values w(x) = 1/sqrt(1-x^2).

    Notes
    -----
    The weight is undefined at x = ±1. Points near the boundary will
    have very large weights.

    Examples
    --------
    >>> chebyshev_polynomial_t_weight(torch.tensor([0.0]))
    tensor([1.])
    """
    domain = ChebyshevPolynomialT.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating ChebyshevPolynomialT weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}]. Results may be NaN or complex.",
            stacklevel=2,
        )

    return 1.0 / torch.sqrt(1.0 - x**2)
