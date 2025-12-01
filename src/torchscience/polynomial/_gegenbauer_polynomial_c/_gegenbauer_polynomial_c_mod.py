from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
from ._gegenbauer_polynomial_c_divmod import gegenbauer_polynomial_c_divmod


def gegenbauer_polynomial_c_mod(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Divide two Gegenbauer series, returning remainder only.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Dividend.
    b : GegenbauerPolynomialC
        Divisor.

    Returns
    -------
    GegenbauerPolynomialC
        Remainder.
    """
    _, r = gegenbauer_polynomial_c_divmod(a, b)
    return r
