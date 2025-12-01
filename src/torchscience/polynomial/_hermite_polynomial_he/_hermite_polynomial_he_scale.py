"""Scale a Probabilists' Hermite series by a scalar."""

from __future__ import annotations

from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_scale(
    a: HermitePolynomialHe,
    scalar: Tensor,
) -> HermitePolynomialHe:
    """Scale a Probabilists' Hermite series by a scalar.

    Parameters
    ----------
    a : HermitePolynomialHe
        Series to scale.
    scalar : Tensor
        Scalar multiplier.

    Returns
    -------
    HermitePolynomialHe
        Scaled series scalar * a.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0, 2.0, 3.0]))
    >>> b = hermite_polynomial_he_scale(a, torch.tensor(2.0))
    >>> b.coeffs
    tensor([2., 4., 6.])
    """
    return HermitePolynomialHe(coeffs=a.coeffs * scalar)
