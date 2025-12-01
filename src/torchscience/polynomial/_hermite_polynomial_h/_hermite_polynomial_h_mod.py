from ._hermite_polynomial_h import HermitePolynomialH
from ._hermite_polynomial_h_divmod import hermite_polynomial_h_divmod


def hermite_polynomial_h_mod(
    a: HermitePolynomialH,
    b: HermitePolynomialH,
) -> HermitePolynomialH:
    """Divide two Physicists' Hermite series, returning remainder only.

    Parameters
    ----------
    a : HermitePolynomialH
        Dividend.
    b : HermitePolynomialH
        Divisor.

    Returns
    -------
    HermitePolynomialH
        Remainder.
    """
    _, r = hermite_polynomial_h_divmod(a, b)
    return r
