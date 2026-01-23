"""Find roots of Euler polynomial series."""

from torch import Tensor

from ._euler_polynomial_e import EulerPolynomialE
from ._euler_polynomial_e_to_polynomial import (
    euler_polynomial_e_to_polynomial,
)


def euler_polynomial_e_roots(a: EulerPolynomialE) -> Tensor:
    """Find roots of an Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Series to find roots of.

    Returns
    -------
    Tensor
        Complex roots of the polynomial.

    Notes
    -----
    Converts to standard polynomial and finds roots via companion matrix.
    """
    from torchscience.polynomial._polynomial._polynomial_roots import (
        polynomial_roots,
    )

    # Convert to standard polynomial
    p = euler_polynomial_e_to_polynomial(a)

    # Find roots of standard polynomial
    return polynomial_roots(p)
