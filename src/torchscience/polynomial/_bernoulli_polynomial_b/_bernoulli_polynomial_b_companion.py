"""Companion matrix for Bernoulli polynomial series."""

from torch import Tensor

from ._bernoulli_polynomial_b import BernoulliPolynomialB
from ._bernoulli_polynomial_b_to_polynomial import (
    bernoulli_polynomial_b_to_polynomial,
)


def bernoulli_polynomial_b_companion(a: BernoulliPolynomialB) -> Tensor:
    """Return companion matrix for a Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Bernoulli polynomial series.

    Returns
    -------
    Tensor
        Companion matrix.

    Notes
    -----
    Converts to standard polynomial and computes companion matrix.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 1.0]))
    >>> C = bernoulli_polynomial_b_companion(a)
    """
    from torchscience.polynomial._polynomial._polynomial_companion import (
        polynomial_companion,
    )

    # Convert to standard polynomial
    p = bernoulli_polynomial_b_to_polynomial(a)

    # Get companion matrix
    return polynomial_companion(p)
