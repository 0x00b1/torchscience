import warnings

import torch
from torch import Tensor

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_weight(
    x: Tensor,
) -> Tensor:
    """Compute Laguerre weight function.

    The weight function is w(x) = exp(-x), which appears in the orthogonality
    relation for Laguerre polynomials:

        integral_{0}^{∞} L_m(x) L_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate weight.

    Returns
    -------
    Tensor
        Weight values w(x) = exp(-x).

    Notes
    -----
    The Laguerre polynomials are orthogonal on [0, ∞) with respect to
    the weight function w(x) = exp(-x).

    The integral of L_n(x)^2 * exp(-x) from 0 to ∞ equals 1 for all n.

    Examples
    --------
    >>> laguerre_polynomial_l_weight(torch.tensor([0.0, 1.0, 2.0]))
    tensor([1.0000, 0.3679, 0.1353])
    """
    domain = LaguerrePolynomialL.DOMAIN

    if (x < domain[0]).any():
        warnings.warn(
            f"Evaluating LaguerrePolynomialL weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}).",
            stacklevel=2,
        )

    return torch.exp(-x)
