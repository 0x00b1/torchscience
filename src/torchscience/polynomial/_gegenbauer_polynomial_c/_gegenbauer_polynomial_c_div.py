from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
from ._gegenbauer_polynomial_c_divmod import gegenbauer_polynomial_c_divmod


def gegenbauer_polynomial_c_div(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Divide two Gegenbauer series, returning quotient only.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Dividend.
    b : GegenbauerPolynomialC
        Divisor.

    Returns
    -------
    GegenbauerPolynomialC
        Quotient.
    """
    q, _ = gegenbauer_polynomial_c_divmod(a, b)
    return q
