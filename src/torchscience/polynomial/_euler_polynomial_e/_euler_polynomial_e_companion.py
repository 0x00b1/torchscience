"""Companion matrix for Euler polynomial series."""

from torch import Tensor

from ._euler_polynomial_e import EulerPolynomialE
from ._euler_polynomial_e_to_polynomial import (
    euler_polynomial_e_to_polynomial,
)


def euler_polynomial_e_companion(a: EulerPolynomialE) -> Tensor:
    """Return companion matrix for an Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Euler polynomial series.

    Returns
    -------
    Tensor
        Companion matrix.

    Notes
    -----
    Converts to standard polynomial and computes companion matrix.
    """
    from torchscience.polynomial._polynomial._polynomial_companion import (
        polynomial_companion,
    )

    # Convert to standard polynomial
    p = euler_polynomial_e_to_polynomial(a)

    # Get companion matrix
    return polynomial_companion(p)
