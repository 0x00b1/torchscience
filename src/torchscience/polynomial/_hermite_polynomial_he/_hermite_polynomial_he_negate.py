"""Negate a Probabilists' Hermite series."""

from __future__ import annotations

import torch

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)


def hermite_polynomial_he_negate(
    a: HermitePolynomialHe,
) -> HermitePolynomialHe:
    """Negate a Probabilists' Hermite series.

    Parameters
    ----------
    a : HermitePolynomialHe
        Series to negate.

    Returns
    -------
    HermitePolynomialHe
        Negated series -a.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0, -2.0, 3.0]))
    >>> b = hermite_polynomial_he_negate(a)
    >>> b
    HermitePolynomialHe(tensor([-1.,  2., -3.]))
    """
    # Convert to plain tensor to avoid operator interception
    result = -a.as_subclass(torch.Tensor)
    return hermite_polynomial_he(result)
