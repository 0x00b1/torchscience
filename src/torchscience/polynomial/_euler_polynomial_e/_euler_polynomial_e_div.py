"""Integer division of Euler polynomial series."""

from ._euler_polynomial_e import EulerPolynomialE
from ._euler_polynomial_e_divmod import euler_polynomial_e_divmod


def euler_polynomial_e_div(
    a: EulerPolynomialE,
    b: EulerPolynomialE,
) -> EulerPolynomialE:
    """Integer division of Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Dividend.
    b : EulerPolynomialE
        Divisor.

    Returns
    -------
    EulerPolynomialE
        Quotient a // b.
    """
    q, _ = euler_polynomial_e_divmod(a, b)
    return q
