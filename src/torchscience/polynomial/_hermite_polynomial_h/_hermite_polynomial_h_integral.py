from torch import Tensor

from ._hermite_polynomial_h import HermitePolynomialH
from ._hermite_polynomial_h_antiderivative import (
    hermite_polynomial_h_antiderivative,
)
from ._hermite_polynomial_h_evaluate import hermite_polynomial_h_evaluate


def hermite_polynomial_h_integral(
    a: HermitePolynomialH,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Physicists' Hermite series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : HermitePolynomialH
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

    Unlike bounded domain polynomials, Hermite polynomials have
    unbounded domain (-inf, inf), so no domain warning is issued.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0]))  # constant 1
    >>> hermite_polynomial_h_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    # No domain warning for Hermite polynomials since domain is (-inf, inf)

    # Compute antiderivative with C=0
    antideriv = hermite_polynomial_h_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = hermite_polynomial_h_evaluate(antideriv, upper)
    f_lower = hermite_polynomial_h_evaluate(antideriv, lower)

    return f_upper - f_lower
