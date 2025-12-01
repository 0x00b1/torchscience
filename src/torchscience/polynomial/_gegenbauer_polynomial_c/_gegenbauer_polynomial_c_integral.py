import warnings

from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
from ._gegenbauer_polynomial_c_antiderivative import (
    gegenbauer_polynomial_c_antiderivative,
)
from ._gegenbauer_polynomial_c_evaluate import gegenbauer_polynomial_c_evaluate


def gegenbauer_polynomial_c_integral(
    a: GegenbauerPolynomialC,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    """Compute definite integral of Gegenbauer series.

    Computes integral_{lower}^{upper} a(x) dx.

    Parameters
    ----------
    a : GegenbauerPolynomialC
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
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0]), torch.tensor(1.0))  # constant 1
    >>> gegenbauer_polynomial_c_integral(a, torch.tensor(-1.0), torch.tensor(1.0))
    tensor(2.)
    """
    domain = GegenbauerPolynomialC.DOMAIN

    if (lower < domain[0]).any() or (upper > domain[1]).any():
        warnings.warn(
            f"Integration bounds extend outside natural domain "
            f"[{domain[0]}, {domain[1]}] for GegenbauerPolynomialC. "
            f"Results may be numerically unstable.",
            stacklevel=2,
        )

    # Compute antiderivative with C=0
    antideriv = gegenbauer_polynomial_c_antiderivative(a, constant=0.0)

    # Evaluate at endpoints
    f_upper = gegenbauer_polynomial_c_evaluate(antideriv, upper)
    f_lower = gegenbauer_polynomial_c_evaluate(antideriv, lower)

    return f_upper - f_lower
