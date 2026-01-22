import torch
from torch import Tensor

from ._chebyshev_polynomial_w import (
    ChebyshevPolynomialW,
    chebyshev_polynomial_w,
)


def chebyshev_polynomial_w_scale(
    a: ChebyshevPolynomialW,
    scalar: Tensor,
) -> ChebyshevPolynomialW:
    """Scale a Chebyshev W series by a scalar.

    Parameters
    ----------
    a : ChebyshevPolynomialW
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    ChebyshevPolynomialW
        Scaled series scalar * a.

    Examples
    --------
    >>> a = chebyshev_polynomial_w(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_w_scale(a, torch.tensor(2.0))
    >>> b
    ChebyshevPolynomialW(tensor([2., 4., 6.]))
    """
    result = a.as_subclass(torch.Tensor) * scalar
    return chebyshev_polynomial_w(result)
