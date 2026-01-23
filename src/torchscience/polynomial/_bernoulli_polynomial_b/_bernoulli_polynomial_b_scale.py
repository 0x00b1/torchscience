"""Scale Bernoulli polynomial series by a scalar."""

import torch
from torch import Tensor

from ._bernoulli_polynomial_b import (
    BernoulliPolynomialB,
    bernoulli_polynomial_b,
)


def bernoulli_polynomial_b_scale(
    a: BernoulliPolynomialB,
    scalar: Tensor,
) -> BernoulliPolynomialB:
    """Scale Bernoulli polynomial series by a scalar.

    Parameters
    ----------
    a : BernoulliPolynomialB
        Series to scale.
    scalar : Tensor
        Scalar value to multiply by.

    Returns
    -------
    BernoulliPolynomialB
        Scaled series a * scalar.

    Examples
    --------
    >>> a = bernoulli_polynomial_b(torch.tensor([1.0, 2.0]))
    >>> bernoulli_polynomial_b_scale(a, torch.tensor(3.0))
    BernoulliPolynomialB(tensor([3., 6.]))
    """
    coeffs = a.as_subclass(torch.Tensor)
    # Ensure scalar is a tensor
    if not isinstance(scalar, Tensor):
        scalar = torch.as_tensor(
            scalar, dtype=coeffs.dtype, device=coeffs.device
        )
    return bernoulli_polynomial_b(coeffs * scalar)
