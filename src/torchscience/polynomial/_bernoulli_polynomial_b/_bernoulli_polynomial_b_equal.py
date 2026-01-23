"""Equality comparison for Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import BernoulliPolynomialB
from ._bernoulli_polynomial_b_trim import bernoulli_polynomial_b_trim


def bernoulli_polynomial_b_equal(
    a: BernoulliPolynomialB,
    b: BernoulliPolynomialB,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check if two Bernoulli polynomial series are equal.

    Parameters
    ----------
    a : BernoulliPolynomialB
        First series.
    b : BernoulliPolynomialB
        Second series.
    rtol : float, optional
        Relative tolerance. Default is 1e-5.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.

    Returns
    -------
    bool
        True if series are approximately equal.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
    >>> b = bernoulli_polynomial_b(torch.tensor([1.0, 2.0, 0.0]))
    >>> bernoulli_polynomial_b_equal(a, b)
    True
    """
    # Trim both series
    a_trimmed = bernoulli_polynomial_b_trim(a, tol=atol)
    b_trimmed = bernoulli_polynomial_b_trim(b, tol=atol)

    a_coeffs = a_trimmed.as_subclass(torch.Tensor)
    b_coeffs = b_trimmed.as_subclass(torch.Tensor)

    # Must have same shape after trimming
    if a_coeffs.shape != b_coeffs.shape:
        return False

    return torch.allclose(a_coeffs, b_coeffs, rtol=rtol, atol=atol)
