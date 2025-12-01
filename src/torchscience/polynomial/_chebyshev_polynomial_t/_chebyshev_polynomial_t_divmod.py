from typing import Tuple

from torchscience.polynomial._polynomial import polynomial_divmod

from ._chebyshev_polynomial_t import ChebyshevPolynomialT
from ._chebyshev_polynomial_t_to_polynomial import (
    chebyshev_polynomial_t_to_polynomial,
)
from ._polynomial_to_chebyshev_polynomial_t import (
    polynomial_to_chebyshev_polynomial_t,
)


def chebyshev_polynomial_t_divmod(
    a: ChebyshevPolynomialT,
    b: ChebyshevPolynomialT,
) -> Tuple[ChebyshevPolynomialT, ChebyshevPolynomialT]:
    """Divide two Chebyshev series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Dividend.
    b : ChebyshevPolynomialT
        Divisor.

    Returns
    -------
    Tuple[ChebyshevPolynomialT, ChebyshevPolynomialT]
        (quotient, remainder)

    Notes
    -----
    Converts to power basis, performs division, converts back.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_t(torch.tensor([1.0, 1.0]))
    >>> q, r = chebyshev_polynomial_t_divmod(a, b)
    """
    # Convert to power basis
    a_poly = chebyshev_polynomial_t_to_polynomial(a)
    b_poly = chebyshev_polynomial_t_to_polynomial(b)

    # Divide in power basis
    q_poly, r_poly = polynomial_divmod(a_poly, b_poly)

    # Convert back to Chebyshev
    q = polynomial_to_chebyshev_polynomial_t(q_poly)
    r = polynomial_to_chebyshev_polynomial_t(r_poly)

    return q, r
