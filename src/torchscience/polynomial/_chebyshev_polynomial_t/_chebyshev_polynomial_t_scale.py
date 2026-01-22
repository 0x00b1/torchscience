import torch
from torch import Tensor

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)


def chebyshev_polynomial_t_scale(
    a: ChebyshevPolynomialT,
    scalar: Tensor,
) -> ChebyshevPolynomialT:
    """Scale a Chebyshev series by a scalar.

    Parameters
    ----------
    a : ChebyshevPolynomialT
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    ChebyshevPolynomialT
        Scaled series scalar * a.

    Examples
    --------
    >>> a = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = chebyshev_polynomial_t_scale(a, torch.tensor(2.0))
    >>> b
    ChebyshevPolynomialT(tensor([2., 4., 6.]))
    """
    return chebyshev_polynomial_t(a.as_subclass(torch.Tensor) * scalar)
