from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_divmod import laguerre_polynomial_l_divmod


def laguerre_polynomial_l_mod(
    a: LaguerrePolynomialL,
    b: LaguerrePolynomialL,
) -> LaguerrePolynomialL:
    """Divide two Laguerre series, returning remainder only.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Dividend.
    b : LaguerrePolynomialL
        Divisor.

    Returns
    -------
    LaguerrePolynomialL
        Remainder.
    """
    _, r = laguerre_polynomial_l_divmod(a, b)
    return r
