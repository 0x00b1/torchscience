import torch
from torch import Tensor

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)


def legendre_polynomial_p_scale(
    a: LegendrePolynomialP,
    scalar: Tensor,
) -> LegendrePolynomialP:
    """Scale a Legendre series by a scalar.

    Parameters
    ----------
    a : LegendrePolynomialP
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    LegendrePolynomialP
        Scaled series scalar * a.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = legendre_polynomial_p_scale(a, torch.tensor(2.0))
    >>> b
    LegendrePolynomialP(tensor([2., 4., 6.]))
    """
    return legendre_polynomial_p(a.as_subclass(torch.Tensor) * scalar)
