"""Modulo of Euler polynomial series."""

from ._euler_polynomial_e import EulerPolynomialE
from ._euler_polynomial_e_divmod import euler_polynomial_e_divmod


def euler_polynomial_e_mod(
    a: EulerPolynomialE,
    b: EulerPolynomialE,
) -> EulerPolynomialE:
    """Modulo of Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Dividend.
    b : EulerPolynomialE
        Divisor.

    Returns
    -------
    EulerPolynomialE
        Remainder a % b.
    """
    _, r = euler_polynomial_e_divmod(a, b)
    return r
