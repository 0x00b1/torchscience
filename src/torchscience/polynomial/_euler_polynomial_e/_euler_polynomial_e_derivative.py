"""Derivative of Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def euler_polynomial_e_derivative(
    a: EulerPolynomialE,
    order: int = 1,
) -> EulerPolynomialE:
    r"""Compute derivative of Euler polynomial series.

    Uses the fundamental property of Euler polynomials:

    .. math::

        E'_n(x) = n \cdot E_{n-1}(x)

    For f(x) = sum_{k=0}^{n} c[k] * E_k(x), the derivative is:

    .. math::

        f'(x) = \sum_{k=1}^{n} c[k] \cdot k \cdot E_{k-1}(x)
              = \sum_{j=0}^{n-1} c[j+1] \cdot (j+1) \cdot E_j(x)

    Parameters
    ----------
    a : EulerPolynomialE
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    EulerPolynomialE
        Derivative series.

    Raises
    ------
    ValueError
        If order is negative.

    Notes
    -----
    The degree decreases by 1 for each differentiation.
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    coeffs = a.as_subclass(torch.Tensor)

    if order == 0:
        return euler_polynomial_e(coeffs.clone())

    # Apply derivative 'order' times
    for _ in range(order):
        n = coeffs.shape[-1]

        if n <= 1:
            # Derivative of constant is zero
            result_shape = list(coeffs.shape)
            result_shape[-1] = 1
            coeffs = torch.zeros(
                result_shape, dtype=coeffs.dtype, device=coeffs.device
            )
            continue

        # For f(x) = sum c[k] * E_k(x)
        # f'(x) = sum c[k] * k * E_{k-1}(x)
        #       = sum_{j=0}^{n-2} c[j+1] * (j+1) * E_j(x)
        new_n = n - 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        new_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # k goes from 1 to n-1, j = k-1 goes from 0 to n-2
        for j in range(new_n):
            k = j + 1
            new_coeffs[..., j] = coeffs[..., k] * k

        coeffs = new_coeffs

    return euler_polynomial_e(coeffs)
