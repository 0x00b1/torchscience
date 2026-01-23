"""Power of Bernoulli polynomial series."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)
from ._bernoulli_polynomial_b_multiply import bernoulli_polynomial_b_multiply


def bernoulli_polynomial_b_pow(
    a: BernoulliPolynomialB,
    n: int,
) -> BernoulliPolynomialB:
    """Raise Bernoulli polynomial series to a power.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to raise to power.
    n : int
        Non-negative integer power.

    Returns
    -------
    BernoulliPolynomialB
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 1.0]))
    >>> bernoulli_polynomial_b_pow(a, 2)  # (B_0 + B_1)^2
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"Power must be a non-negative integer, got {n}")

    if n == 0:
        # Return 1 (which is B_0)
        result_shape = list(a.shape)
        result_shape[-1] = 1
        coeffs = torch.ones(result_shape, dtype=a.dtype, device=a.device)
        return bernoulli_polynomial_b(coeffs)

    if n == 1:
        return bernoulli_polynomial_b(a.as_subclass(torch.Tensor).clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = bernoulli_polynomial_b(
                    base.as_subclass(torch.Tensor).clone()
                )
            else:
                result = bernoulli_polynomial_b_multiply(result, base)
        base = bernoulli_polynomial_b_multiply(base, base)
        n //= 2

    return result
