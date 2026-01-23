"""Trim trailing zeros from Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_trim(
    a: BernoulliPolynomialB,
    tol: float = 0.0,
) -> BernoulliPolynomialB:
    """Trim trailing near-zero coefficients from Bernoulli polynomial series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to trim.
    tol : float, optional
        Tolerance for considering a coefficient as zero. Default is 0.0.

    Returns
    -------
    BernoulliPolynomialB
        Trimmed series.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 0.0, 0.0]))
    >>> bernoulli_polynomial_b_trim(a)
    BernoulliPolynomialB(tensor([1., 2.]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Find last non-zero coefficient
    # For batched, consider all batches
    last_nonzero = 0
    for k in range(n):
        if torch.any(torch.abs(coeffs[..., k]) > tol):
            last_nonzero = k

    # Keep at least one coefficient
    new_n = max(last_nonzero + 1, 1)

    return bernoulli_polynomial_b(coeffs[..., :new_n].clone())
