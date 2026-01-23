"""Scale Euler polynomial series by a scalar."""

import torch
from torch import Tensor

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def euler_polynomial_e_scale(
    a: EulerPolynomialE,
    scalar: Tensor,
) -> EulerPolynomialE:
    """Scale Euler polynomial series by a scalar.

    Parameters
    ----------
    a : EulerPolynomialE
        Series to scale.
    scalar : Tensor
        Scalar value to multiply by.

    Returns
    -------
    EulerPolynomialE
        Scaled series a * scalar.
    """
    coeffs = a.as_subclass(torch.Tensor)
    # Ensure scalar is a tensor
    if not isinstance(scalar, Tensor):
        scalar = torch.as_tensor(
            scalar, dtype=coeffs.dtype, device=coeffs.device
        )
    return euler_polynomial_e(coeffs * scalar)
