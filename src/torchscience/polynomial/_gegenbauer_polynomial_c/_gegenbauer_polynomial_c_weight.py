import warnings

import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_weight(
    x: Tensor,
    lambda_: Tensor,
) -> Tensor:
    """Compute Gegenbauer weight function.

    The weight function is w(x) = (1 - x^2)^{lambda - 1/2}, which appears in
    the orthogonality relation for Gegenbauer polynomials:

        integral_{-1}^{1} C_m^{lambda}(x) C_n^{lambda}(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate weight.
    lambda_ : Tensor
        Parameter lambda > -1/2.

    Returns
    -------
    Tensor
        Weight values w(x) = (1 - x^2)^{lambda - 1/2}.

    Notes
    -----
    For lambda = 1/2, the weight is constant (Legendre case).
    For lambda = 1, w(x) = sqrt(1-x^2) (Chebyshev U case).
    For lambda approaching 0, w(x) = 1/sqrt(1-x^2) (Chebyshev T limit).

    The weight is only defined on the open interval (-1, 1).
    At the endpoints x = ±1:
    - For lambda > 1/2: w(±1) = 0
    - For lambda = 1/2: w(±1) = 1
    - For lambda < 1/2: w(±1) = infinity

    Examples
    --------
    >>> gegenbauer_polynomial_c_weight(torch.tensor([0.0, 0.5]), torch.tensor(1.0))
    tensor([1.0000, 0.8660])
    """
    domain = GegenbauerPolynomialC.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating GegenbauerPolynomialC weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}].",
            stacklevel=2,
        )

    # Ensure lambda_ is broadcastable
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(lambda_, dtype=x.dtype, device=x.device)

    # w(x) = (1 - x^2)^{lambda - 1/2}
    exponent = lambda_ - 0.5
    base = 1.0 - x * x

    # Handle edge cases where base might be negative due to numerical errors
    base = torch.clamp(base, min=0.0)

    return torch.pow(base, exponent)
