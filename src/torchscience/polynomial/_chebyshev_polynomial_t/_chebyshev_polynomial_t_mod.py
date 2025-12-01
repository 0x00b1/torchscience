from ._chebyshev_polynomial_t import ChebyshevPolynomialT
from ._chebyshev_polynomial_t_divmod import chebyshev_polynomial_t_divmod


def chebyshev_polynomial_t_mod(
    a: ChebyshevPolynomialT,
    b: ChebyshevPolynomialT,
) -> ChebyshevPolynomialT:
    """Divide two Chebyshev series, returning remainder only.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Dividend.
    b : ChebyshevPolynomialT
        Divisor.

    Returns
    -------
    ChebyshevPolynomialT
        Remainder.
    """
    _, r = chebyshev_polynomial_t_divmod(a, b)
    return r
