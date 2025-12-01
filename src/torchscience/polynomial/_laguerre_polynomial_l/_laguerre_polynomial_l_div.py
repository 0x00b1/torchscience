from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_divmod import laguerre_polynomial_l_divmod


def laguerre_polynomial_l_div(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Divide two Laguerre series, returning quotient only.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Dividend.
    b : LaguerrePolynomialL
        Divisor.

    Returns
    -------
    LaguerrePolynomialL
        Quotient.
    """
    q, _ = laguerre_polynomial_l_divmod(a, b)
    return q
