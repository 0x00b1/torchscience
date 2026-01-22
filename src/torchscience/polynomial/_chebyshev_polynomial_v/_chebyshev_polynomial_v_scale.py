import torch
from torch import Tensor

from ._chebyshev_polynomial_v import (
    ChebyshevPolynomialV,
    chebyshev_polynomial_v,
)


def chebyshev_polynomial_v_scale(
    a: ChebyshevPolynomialV,
    scalar: Tensor,
) -> ChebyshevPolynomialV:
    """Scale a Chebyshev V series by a scalar.

    Parameters
    ----------
    a : ChebyshevPolynomialV
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    ChebyshevPolynomialV
        Scaled series scalar * a.

    Examples
    --------
    >>> a = chebyshev_polynomial_v(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_v_scale(a, torch.tensor(2.0))
    >>> b
    ChebyshevPolynomialV(tensor([2., 4., 6.]))
    """
    return chebyshev_polynomial_v(a.as_subclass(torch.Tensor) * scalar)
