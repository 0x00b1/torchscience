from typing import Tuple

from torchscience.polynomial._polynomial import polynomial_divmod

from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_to_polynomial import (
    chebyshev_polynomial_w_to_polynomial,
)
from ._polynomial_to_chebyshev_polynomial_w import (
    polynomial_to_chebyshev_polynomial_w,
)


def chebyshev_polynomial_w_divmod(
    a: ChebyshevPolynomialW,
    b: ChebyshevPolynomialW,
) -> Tuple[ChebyshevPolynomialW, ChebyshevPolynomialW]:
    """Divide two Chebyshev W series with remainder.

    Returns quotient q and remainder r such that a = b*q + r.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Dividend.
    b : ChebyshevPolynomialW
        Divisor.

    Returns
    -------
    Tuple[ChebyshevPolynomialW, ChebyshevPolynomialW]
        (quotient, remainder)

    Notes
    -----
    Converts to power basis, performs division, converts back.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_w(torch.tensor([1.0, 1.0]))
    >>> q, r = chebyshev_polynomial_w_divmod(a, b)
    """
    # Convert to power basis
    a_poly = chebyshev_polynomial_w_to_polynomial(a)
    b_poly = chebyshev_polynomial_w_to_polynomial(b)

    # Divide in power basis
    q_poly, r_poly = polynomial_divmod(a_poly, b_poly)

    # Convert back to Chebyshev W
    q = polynomial_to_chebyshev_polynomial_w(q_poly)
    r = polynomial_to_chebyshev_polynomial_w(r_poly)

    return q, r
