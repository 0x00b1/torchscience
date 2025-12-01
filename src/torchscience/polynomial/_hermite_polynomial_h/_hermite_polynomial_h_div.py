from ._hermite_polynomial_h import HermitePolynomialH
from ._hermite_polynomial_h_divmod import hermite_polynomial_h_divmod


def hermite_polynomial_h_div(
    a: HermitePolynomialH,
    b: HermitePolynomialH,
) -> HermitePolynomialH:
    """Divide two Physicists' Hermite series, returning quotient only.

    Parameters
    ----------
    a : HermitePolynomialH
        Dividend.
    b : HermitePolynomialH
        Divisor.

    Returns
    -------
    HermitePolynomialH
        Quotient.
    """
    q, _ = hermite_polynomial_h_divmod(a, b)
    return q
