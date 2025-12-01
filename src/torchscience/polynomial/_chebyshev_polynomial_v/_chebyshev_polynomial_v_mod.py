from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_divmod import chebyshev_polynomial_v_divmod


def chebyshev_polynomial_v_mod(
    a: ChebyshevPolynomialV,
    b: ChebyshevPolynomialV,
) -> ChebyshevPolynomialV:
    """Divide two Chebyshev V series, returning remainder only.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Dividend.
    b : ChebyshevPolynomialV
        Divisor.

    Returns
    -------
    ChebyshevPolynomialV
        Remainder.
    """
    _, r = chebyshev_polynomial_v_divmod(a, b)
    return r
