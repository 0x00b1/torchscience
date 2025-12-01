"""Derivative of Probabilists' Hermite series."""

from __future__ import annotations

import torch

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_derivative(
    a: HermitePolynomialHe,
    order: int = 1,
) -> HermitePolynomialHe:
    """Compute derivative of Probabilists' Hermite series.

    Uses the derivative formula for Hermite_e polynomials:
        d/dx He_n(x) = n * He_{n-1}(x)

    Parameters
    ----------
    a : HermitePolynomialHe
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    HermitePolynomialHe
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    For a Hermite_e series f(x) = sum_{k=0}^{n} c_k He_k(x),
    the derivative is:
        f'(x) = sum_{k=0}^{n} c_k * k * He_{k-1}(x)
              = sum_{j=0}^{n-1} ((j+1) * c_{j+1}) * He_j(x)

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([0.0, 1.0]))  # He_1 = x
    >>> da = hermite_polynomial_he_derivative(a)
    >>> da.coeffs  # d/dx He_1 = 1*He_0 = 1
    tensor([1.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return HermitePolynomialHe(coeffs=a.coeffs.clone())

    coeffs = a.coeffs.clone()
    n = coeffs.shape[-1]

    # Apply derivative 'order' times
    for _ in range(order):
        if n <= 1:
            # Derivative of constant is zero
            result_shape = list(coeffs.shape)
            result_shape[-1] = 1
            coeffs = torch.zeros(
                result_shape, dtype=coeffs.dtype, device=coeffs.device
            )
            n = 1
            continue

        # Result has n-1 coefficients (degree decreases by 1)
        new_n = n - 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        der = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # d/dx He_k = k * He_{k-1}
        # So coefficient of He_j in derivative is (j+1) * c_{j+1}
        for j in range(new_n):
            der[..., j] = (j + 1) * coeffs[..., j + 1]

        coeffs = der
        n = new_n

    return HermitePolynomialHe(coeffs=coeffs)
