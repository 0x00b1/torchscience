import warnings

from torch import Tensor

from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_antiderivative import (
    laguerre_polynomial_l_antiderivative,
)
from ._laguerre_polynomial_l_evaluate import laguerre_polynomial_l_evaluate


def laguerre_polynomial_l_integral(
    a: LaguerrePolynomialL,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Laguerre series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : LaguerrePolynomialL
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
    >>> a = laguerre_polynomial_l(torch.tensor([1.0]))  # constant 1
    >>> laguerre_polynomial_l_integral(a, torch.tensor(0.0), torch.tensor(1.0))
    tensor(1.)
    """
    domain = LaguerrePolynomialL.DOMAIN

    if (lower < domain[0]).any():
        warnings.warn(
            f"Integration lower bound extends outside natural domain "
            f"[{domain[0]}, {domain[1]}) for LaguerrePolynomialL. "
            f"Results may be numerically unstable.",
            stacklevel=2,
        )

    # Compute antiderivative with C=0
    antideriv = laguerre_polynomial_l_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = laguerre_polynomial_l_evaluate(antideriv, upper)
    f_lower = laguerre_polynomial_l_evaluate(antideriv, lower)

    return f_upper - f_lower
