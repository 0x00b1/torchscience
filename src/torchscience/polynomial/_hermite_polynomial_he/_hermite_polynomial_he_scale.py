"""Scale a Probabilists' Hermite series by a scalar."""

from __future__ import annotations

import torch
from torch import Tensor

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)


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
    >>> b
    HermitePolynomialHe(tensor([2., 4., 6.]))
    """
    # Convert to plain tensor for multiplication to avoid operator interception
    result = a.as_subclass(torch.Tensor) * scalar
    return hermite_polynomial_he(result)
