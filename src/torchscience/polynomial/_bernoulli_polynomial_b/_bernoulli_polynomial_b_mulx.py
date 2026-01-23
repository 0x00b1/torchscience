"""Multiply Bernoulli polynomial series by x."""

import torch

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
)
from ._bernoulli_polynomial_b_to_polynomial import (
    bernoulli_polynomial_b_to_polynomial,
)
from ._polynomial_to_bernoulli_polynomial_b import (
    polynomial_to_bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_mulx(
    a: BernoulliPolynomialB,
) -> BernoulliPolynomialB:
    """Multiply Bernoulli polynomial series by x.

    Computes f(x) * x where f(x) is represented by the input series.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Input series.

    Returns
    -------
    BernoulliPolynomialB
        Product a(x) * x.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0]))  # B_0(x) = 1
    >>> ax = bernoulli_polynomial_b_mulx(a)  # Should give x in Bernoulli basis
    """
    from torchscience.polynomial._polynomial import Polynomial

    # Convert to standard polynomial
    p = bernoulli_polynomial_b_to_polynomial(a)
    p_coeffs = p.as_subclass(torch.Tensor)

    # Multiply by x: shift coefficients
    n = p_coeffs.shape[-1]
    new_shape = list(p_coeffs.shape)
    new_shape[-1] = n + 1

    new_coeffs = torch.zeros(
        new_shape, dtype=p_coeffs.dtype, device=p_coeffs.device
    )
    new_coeffs[..., 1:] = p_coeffs

    # Convert back to Bernoulli basis
    p_new = Polynomial(new_coeffs)
    return polynomial_to_bernoulli_polynomial_b(p_new)
