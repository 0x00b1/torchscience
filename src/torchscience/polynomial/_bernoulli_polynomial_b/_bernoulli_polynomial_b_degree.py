"""Degree of Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import BernoulliPolynomialB


def bernoulli_polynomial_b_degree(a: BernoulliPolynomialB) -> int:
    """Return the degree of a Bernoulli polynomial series.

    The degree is the highest index k with non-zero coefficient c[k].

    Parameters
    ----------
    a : BernoulliPolynomialB
        Bernoulli polynomial series.

    Returns
    -------
    int
        Degree of the series.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 3.0]))
    >>> bernoulli_polynomial_b_degree(a)
    2
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 0.0]))
    >>> bernoulli_polynomial_b_degree(a)
    1
    """
    coeffs = a.as_subclass(torch.Tensor)

    # Find highest non-zero coefficient
    # For batched input, we consider the maximum degree across all batches
    n = coeffs.shape[-1]

    for k in range(n - 1, -1, -1):
        if torch.any(coeffs[..., k] != 0):
            return k

    return 0
