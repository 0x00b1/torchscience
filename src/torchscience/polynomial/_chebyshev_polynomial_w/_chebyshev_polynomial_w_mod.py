from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_divmod import chebyshev_polynomial_w_divmod


def chebyshev_polynomial_w_mod(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
) -> ChebyshevPolynomialW:
    """Divide two Chebyshev W series, returning remainder only.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Dividend.
    b : ChebyshevPolynomialW
        Divisor.

    Returns
    -------
    ChebyshevPolynomialW
        Remainder.
    """
    _, r = chebyshev_polynomial_w_divmod(a, b)
    return r
