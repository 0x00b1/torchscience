"""Divmod of Euler polynomial series."""

from typing import Tuple

from ._euler_polynomial_e import (
    EulerPolynomialE,
)
from ._euler_polynomial_e_to_polynomial import (
    euler_polynomial_e_to_polynomial,
)
from ._polynomial_to_euler_polynomial_e import (
    polynomial_to_euler_polynomial_e,
)


def euler_polynomial_e_divmod(
    a: EulerPolynomialE,
    b: EulerPolynomialE,
) -> Tuple[EulerPolynomialE, EulerPolynomialE]:
    """Divmod of Euler polynomial series.

    Computes quotient and remainder such that a = b * q + r.

    Parameters
    ----------
    a : EulerPolynomialE
        Dividend.
    b : EulerPolynomialE
        Divisor.

    Returns
    -------
    Tuple[EulerPolynomialE, EulerPolynomialE]
        Quotient and remainder (q, r).
    """
    from torchscience.polynomial._polynomial._polynomial_divmod import (
        polynomial_divmod,
    )

    # Convert to standard polynomial basis
    p_a = euler_polynomial_e_to_polynomial(a)
    p_b = euler_polynomial_e_to_polynomial(b)

    # Perform divmod in standard basis
    p_q, p_r = polynomial_divmod(p_a, p_b)

    # Convert back to Euler basis
    q = polynomial_to_euler_polynomial_e(p_q)
    r = polynomial_to_euler_polynomial_e(p_r)

    return q, r
