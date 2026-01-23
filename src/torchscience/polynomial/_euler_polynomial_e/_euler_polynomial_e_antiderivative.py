"""Antiderivative of Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def euler_polynomial_e_antiderivative(
    a: EulerPolynomialE,
    order: int = 1,
    constant: float = 0.0,
) -> EulerPolynomialE:
    r"""Compute antiderivative of Euler polynomial series.

    Uses the inverse of the derivative property:

    .. math::

        \int E_n(x) dx = \frac{E_{n+1}(x)}{n+1} + C

    For f(x) = sum_{k=0}^{n} c[k] * E_k(x), the antiderivative is:

    .. math::

        \int f(x) dx = C \cdot E_0(x) + \sum_{k=0}^{n} \frac{c[k]}{k+1} E_{k+1}(x)

    Parameters
    ----------
    a : EulerPolynomialE
        Series to integrate.
    order : int, optional
        Order of antiderivative. Default is 1.
    constant : float, optional
        Integration constant. Default is 0.0.

    Returns
    -------
    EulerPolynomialE
        Antiderivative series.

    Raises
    ------
    ValueError
        If order is negative.

    Notes
    -----
    The degree increases by 1 for each integration.
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    coeffs = a.as_subclass(torch.Tensor)

    if order == 0:
        return euler_polynomial_e(coeffs.clone())

    # Apply antiderivative 'order' times
    for _ in range(order):
        n = coeffs.shape[-1]

        # New size is n + 1
        new_n = n + 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        new_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # Integration constant goes to E_0 coefficient
        new_coeffs[..., 0] = constant

        # For each c[k] * E_k(x), antiderivative is c[k]/(k+1) * E_{k+1}(x)
        for k in range(n):
            new_coeffs[..., k + 1] = coeffs[..., k] / (k + 1)

        coeffs = new_coeffs
        constant = 0.0  # Only first integration gets the constant

    return euler_polynomial_e(coeffs)
