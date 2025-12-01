from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_divmod import chebyshev_polynomial_w_divmod


def chebyshev_polynomial_w_div(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Divide two Chebyshev W series, returning quotient only.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Dividend.
    b : ChebyshevPolynomialW
        Divisor.

    Returns
    -------
    ChebyshevPolynomialW
        Quotient.
    """
    q, _ = chebyshev_polynomial_w_divmod(a, b)
    return q
