from ._chebyshev_polynomial_t import ChebyshevPolynomialT
from ._chebyshev_polynomial_t_divmod import chebyshev_polynomial_t_divmod


def chebyshev_polynomial_t_div(
    a: ChebyshevPolynomialT,
    b: ChebyshevPolynomialT,
) -> ChebyshevPolynomialT:
    """Divide two Chebyshev series, returning quotient only.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Dividend.
    b : ChebyshevPolynomialT
        Divisor.

    Returns
    -------
    ChebyshevPolynomialT
        Quotient.
    """
    q, _ = chebyshev_polynomial_t_divmod(a, b)
    return q
