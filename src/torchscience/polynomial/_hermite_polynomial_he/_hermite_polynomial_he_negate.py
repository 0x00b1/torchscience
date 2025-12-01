"""Negate a Probabilists' Hermite series."""

from __future__ import annotations

from ._hermite_polynomial_he import HermitePolynomialHe


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
    >>> b.coeffs
    tensor([-1.,  2., -3.])
    """
    return HermitePolynomialHe(coeffs=-a.coeffs)
