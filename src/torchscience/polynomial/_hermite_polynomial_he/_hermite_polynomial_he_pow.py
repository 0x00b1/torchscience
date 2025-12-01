"""Raise Probabilists' Hermite series to an integer power."""

from __future__ import annotations

import torch

from ._hermite_polynomial_he import HermitePolynomialHe
from ._hermite_polynomial_he_multiply import hermite_polynomial_he_multiply


def hermite_polynomial_he_pow(
    a: HermitePolynomialHe,
    n: int,
) -> HermitePolynomialHe:
    """Raise Probabilists' Hermite series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : HermitePolynomialHe
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    HermitePolynomialHe
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0, 1.0]))  # He_0 + He_1
    >>> b = hermite_polynomial_he_pow(a, 2)
    >>> # (He_0 + He_1)^2 using Hermite linearization
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = He_0 = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return HermitePolynomialHe(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            )
        )

    if n == 1:
        return HermitePolynomialHe(coeffs=a.coeffs.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = HermitePolynomialHe(coeffs=base.coeffs.clone())
            else:
                result = hermite_polynomial_he_multiply(result, base)
        base = hermite_polynomial_he_multiply(base, base)
        n //= 2

    return result
