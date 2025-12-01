from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_divmod import legendre_polynomial_p_divmod


def legendre_polynomial_p_div(
    a: LegendrePolynomialP,
    b: LegendrePolynomialP,
) -> LegendrePolynomialP:
    """Divide two Legendre series, returning quotient only.

    Parameters
    ----------
    a : LegendrePolynomialP
        Dividend.
    b : LegendrePolynomialP
        Divisor.

    Returns
    -------
    LegendrePolynomialP
        Quotient.
    """
    q, _ = legendre_polynomial_p_divmod(a, b)
    return q
