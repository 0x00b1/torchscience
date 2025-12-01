import warnings

from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_antiderivative import (
    legendre_polynomial_p_antiderivative,
)
from ._legendre_polynomial_p_evaluate import legendre_polynomial_p_evaluate


def legendre_polynomial_p_integral(
    a: LegendrePolynomialP,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Legendre series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : LegendrePolynomialP
        Series to integrate.
    lower : Tensor
        Lower limit of integration.
    upper : Tensor
        Upper limit of integration.

    Returns
    -------
    Tensor
        Value of definite integral.

    Notes
    -----
    Computed as F(upper) - F(lower) where F is the antiderivative.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0]))  # constant 1
    >>> legendre_polynomial_p_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    domain = LegendrePolynomialP.DOMAIN

    if (lower < domain[0]).any() or (upper > domain[1]).any():
        warnings.warn(
            f"Integration bounds extend outside natural domain "
            f"[{domain[0]}, {domain[1]}] for LegendrePolynomialP. "
            f"Results may be numerically unstable.",
            stacklevel=2,
        )

    # Compute antiderivative with C=0
    antideriv = legendre_polynomial_p_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = legendre_polynomial_p_evaluate(antideriv, upper)
    f_lower = legendre_polynomial_p_evaluate(antideriv, lower)

    return f_upper - f_lower
