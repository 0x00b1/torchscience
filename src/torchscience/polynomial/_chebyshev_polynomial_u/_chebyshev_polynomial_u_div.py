from ._chebyshev_polynomial_u import ChebyshevPolynomialU
from ._chebyshev_polynomial_u_divmod import chebyshev_polynomial_u_divmod


def chebyshev_polynomial_u_div(
    a: ChebyshevPolynomialU,
    b: ChebyshevPolynomialU,
) -> ChebyshevPolynomialU:
    """Divide two Chebyshev U series, returning quotient only.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Dividend.
    b : ChebyshevPolynomialU
        Divisor.

    Returns
    -------
    ChebyshevPolynomialU
        Quotient.
    """
    q, _ = chebyshev_polynomial_u_divmod(a, b)
    return q
