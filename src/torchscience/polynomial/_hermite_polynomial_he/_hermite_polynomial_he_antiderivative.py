"""Antiderivative of Probabilists' Hermite series."""

from __future__ import annotations

import torch

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)
from ._hermite_polynomial_he_evaluate import hermite_polynomial_he_evaluate


def hermite_polynomial_he_antiderivative(
    a: HermitePolynomialHe,
    order: int = 1,
    constant: float = 0.0,
) -> HermitePolynomialHe:
    """Compute antiderivative of Probabilists' Hermite series.

    Uses the integration formula for Hermite_e polynomials:
        integral(He_n(x)) dx = He_{n+1}(x) / (n+1)

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0.

    Parameters
    ----------
    a : HermitePolynomialHe
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    HermitePolynomialHe
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    For Hermite_e polynomials:
        d/dx He_n(x) = n * He_{n-1}(x)

    Therefore:
        integral(He_n(x)) dx = He_{n+1}(x) / (n+1) + C

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0]))  # He_0 = 1
    >>> ia = hermite_polynomial_he_antiderivative(a)
    >>> ia  # integral(1) = x = He_1
    HermitePolynomialHe(tensor([0., 1.]))
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return hermite_polynomial_he(a.clone())

    # Convert to plain tensor for operations
    coeffs = a.as_subclass(torch.Tensor)

    # Apply antiderivative 'order' times
    for i in range(order):
        n = coeffs.shape[-1]

        # Result has n+1 coefficients (degree increases by 1)
        result_shape = list(coeffs.shape)
        result_shape[-1] = n + 1
        i_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # integral(He_k) dx = He_{k+1} / (k+1)
        # So coefficient of He_{k+1} in integral is c_k / (k+1)
        for k in range(n):
            i_coeffs[..., k + 1] = coeffs[..., k] / (k + 1)

        # Set constant of integration so F(0) = constant (for first integration)
        # or F(0) = 0 (for subsequent integrations)
        k_val = constant if i == 0 else 0.0
        temp = hermite_polynomial_he(i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = hermite_polynomial_he_evaluate(temp, x_zero)
        # He_0(x) = 1, so adding delta to i_coeffs[..., 0] shifts F(0) by delta
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return hermite_polynomial_he(coeffs)
