"""Negate Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_negate(
    a: BernoulliPolynomialB,
) -> BernoulliPolynomialB:
    """Negate Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to negate.

    Returns
    -------
    BernoulliPolynomialB
        Negated series -a.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
    >>> bernoulli_polynomial_b_negate(a)
    BernoulliPolynomialB(tensor([-1., -2.]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    return bernoulli_polynomial_b(-coeffs)
