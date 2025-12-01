from typing import Tuple

from torchscience.polynomial._polynomial import polynomial_divmod

from ._chebyshev_polynomial_u import ChebyshevPolynomialU
from ._chebyshev_polynomial_u_to_polynomial import (
    chebyshev_polynomial_u_to_polynomial,
)
from ._polynomial_to_chebyshev_polynomial_u import (
    polynomial_to_chebyshev_polynomial_u,
)


def chebyshev_polynomial_u_divmod(
    a: ChebyshevPolynomialU,
    b: ChebyshevPolynomialU,
) -> Tuple[ChebyshevPolynomialU, ChebyshevPolynomialU]:
    """Divide two Chebyshev U series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : ChebyshevPolynomialU
        Dividend.
    b : ChebyshevPolynomialU
        Divisor.

    Returns
    -------
    Tuple[ChebyshevPolynomialU, ChebyshevPolynomialU]
        (quotient, remainder)

    Notes
    -----
    Converts to power basis, performs division, converts back.

    Examples
    --------
    >>> a = chebyshev_polynomial_u(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_u(torch.tensor([1.0, 1.0]))
    >>> q, r = chebyshev_polynomial_u_divmod(a, b)
    """
    # Convert to power basis
    a_poly = chebyshev_polynomial_u_to_polynomial(a)
    b_poly = chebyshev_polynomial_u_to_polynomial(b)

    # Divide in power basis
    q_poly, r_poly = polynomial_divmod(a_poly, b_poly)

    # Convert back to Chebyshev U
    q = polynomial_to_chebyshev_polynomial_u(q_poly)
    r = polynomial_to_chebyshev_polynomial_u(r_poly)

    return q, r
