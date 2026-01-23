"""Multiply Euler polynomial series by x."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
)
from ._euler_polynomial_e_to_polynomial import (
    euler_polynomial_e_to_polynomial,
)
from ._polynomial_to_euler_polynomial_e import (
    polynomial_to_euler_polynomial_e,
)


def euler_polynomial_e_mulx(a: EulerPolynomialE) -> EulerPolynomialE:
    """Multiply Euler polynomial series by x.

    Computes f(x) * x where f(x) is represented by the input series.

    Parameters
    ----------
    a : EulerPolynomialE
        Input series.

    Returns
    -------
    EulerPolynomialE
        Product a(x) * x.
    """
    from torchscience.polynomial._polynomial import Polynomial

    # Convert to standard polynomial
    p = euler_polynomial_e_to_polynomial(a)
    p_coeffs = p.as_subclass(torch.Tensor)

    # Multiply by x: shift coefficients
    n = p_coeffs.shape[-1]
    new_shape = list(p_coeffs.shape)
    new_shape[-1] = n + 1

    new_coeffs = torch.zeros(
        new_shape, dtype=p_coeffs.dtype, device=p_coeffs.device
    )
    new_coeffs[..., 1:] = p_coeffs

    # Convert back to Euler basis
    p_new = Polynomial(new_coeffs)
    return polynomial_to_euler_polynomial_e(p_new)
