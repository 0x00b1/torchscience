"""Negate Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def euler_polynomial_e_negate(
    a: EulerPolynomialE,
) -> EulerPolynomialE:
    """Negate Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Series to negate.

    Returns
    -------
    EulerPolynomialE
        Negated series -a.
    """
    coeffs = a.as_subclass(torch.Tensor)
    return euler_polynomial_e(-coeffs)
