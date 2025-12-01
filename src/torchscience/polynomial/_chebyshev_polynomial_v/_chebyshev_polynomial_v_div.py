from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_divmod import chebyshev_polynomial_v_divmod


def chebyshev_polynomial_v_div(
    a: ChebyshevPolynomialV,
    b: ChebyshevPolynomialV,
) -> ChebyshevPolynomialV:
    """Divide two Chebyshev V series, returning quotient only.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Dividend.
    b : ChebyshevPolynomialV
        Divisor.

    Returns
    -------
    ChebyshevPolynomialV
        Quotient.
    """
    q, _ = chebyshev_polynomial_v_divmod(a, b)
    return q
