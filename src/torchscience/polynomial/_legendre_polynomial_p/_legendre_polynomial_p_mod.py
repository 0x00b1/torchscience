from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_divmod import legendre_polynomial_p_divmod


def legendre_polynomial_p_mod(
    a: LegendrePolynomialP,
    b: LegendrePolynomialP,
) -> LegendrePolynomialP:
    """Divide two Legendre series, returning remainder only.

    Parameters
    ----------
    a : LegendrePolynomialP
        Dividend.
    b : LegendrePolynomialP
        Divisor.

    Returns
    -------
    LegendrePolynomialP
        Remainder.
    """
    _, r = legendre_polynomial_p_divmod(a, b)
    return r
