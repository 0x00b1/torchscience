import warnings

from torch import Tensor

from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_antiderivative import (
    chebyshev_polynomial_v_antiderivative,
)
from ._chebyshev_polynomial_v_evaluate import chebyshev_polynomial_v_evaluate


def chebyshev_polynomial_v_integral(
    a: ChebyshevPolynomialV,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Chebyshev V series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : ChebyshevPolynomialV
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
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0]))  # constant 1
    >>> chebyshev_polynomial_v_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    domain = ChebyshevPolynomialV.DOMAIN

    if (lower < domain[0]).any() or (upper > domain[1]).any():
        warnings.warn(
            f"Integration bounds extend outside natural domain "
            f"[{domain[0]}, {domain[1]}] for ChebyshevPolynomialV. "
            f"Results may be numerically unstable.",
            stacklevel=2,
        )

    # Compute antiderivative with C=0
    antideriv = chebyshev_polynomial_v_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = chebyshev_polynomial_v_evaluate(antideriv, upper)
    f_lower = chebyshev_polynomial_v_evaluate(antideriv, lower)

    return f_upper - f_lower
