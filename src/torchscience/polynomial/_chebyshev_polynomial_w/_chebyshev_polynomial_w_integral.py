import warnings

from torch import Tensor

from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_antiderivative import (
    chebyshev_polynomial_w_antiderivative,
)
from ._chebyshev_polynomial_w_evaluate import chebyshev_polynomial_w_evaluate


def chebyshev_polynomial_w_integral(
    a: ChebyshevPolynomialW,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Chebyshev W series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : ChebyshevPolynomialW
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
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0]))  # constant 1
    >>> chebyshev_polynomial_w_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    domain = ChebyshevPolynomialW.DOMAIN

    if (lower < domain[0]).any() or (upper > domain[1]).any():
        warnings.warn(
            f"Integration bounds extend outside natural domain "
            f"[{domain[0]}, {domain[1]}] for ChebyshevPolynomialW. "
            f"Results may be numerically unstable.",
            stacklevel=2,
        )

    # Compute antiderivative with C=0
    antideriv = chebyshev_polynomial_w_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = chebyshev_polynomial_w_evaluate(antideriv, upper)
    f_lower = chebyshev_polynomial_w_evaluate(antideriv, lower)

    return f_upper - f_lower
