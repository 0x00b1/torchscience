"""Multiply two Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
)
from ._euler_polynomial_e_to_polynomial import (
    euler_polynomial_e_to_polynomial,
)
from ._polynomial_to_euler_polynomial_e import (
    polynomial_to_euler_polynomial_e,
)


def _polynomial_convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convolve two polynomial coefficient tensors (pure PyTorch)."""
    n_a = a.shape[-1]
    n_b = b.shape[-1]
    n_c = n_a + n_b - 1

    # Handle batch dimensions
    batch_shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])

    # Expand to broadcast shape
    a_expanded = a.expand(*batch_shape, n_a)
    b_expanded = b.expand(*batch_shape, n_b)

    # Result tensor
    result_shape = batch_shape + (n_c,)
    c = torch.zeros(result_shape, dtype=a.dtype, device=a.device)

    # Direct convolution
    for i in range(n_a):
        for j in range(n_b):
            c[..., i + j] = (
                c[..., i + j] + a_expanded[..., i] * b_expanded[..., j]
            )

    return c


def euler_polynomial_e_multiply(
    a: EulerPolynomialE,
    b: EulerPolynomialE,
) -> EulerPolynomialE:
    """Multiply two Euler polynomial series.

    Uses conversion to standard polynomial basis for multiplication.

    Parameters
    ----------
    a : EulerPolynomialE
        First series.
    b : EulerPolynomialE
        Second series.

    Returns
    -------
    EulerPolynomialE
        Product a * b.
    """
    from torchscience.polynomial._polynomial import Polynomial

    # Convert to standard polynomial basis
    p_a = euler_polynomial_e_to_polynomial(a)
    p_b = euler_polynomial_e_to_polynomial(b)

    # Get raw coefficient tensors
    p_a_coeffs = p_a.as_subclass(torch.Tensor)
    p_b_coeffs = p_b.as_subclass(torch.Tensor)

    # Multiply in standard basis using convolution
    p_c_coeffs = _polynomial_convolve(p_a_coeffs, p_b_coeffs)

    # Convert back to Euler basis
    p_c = Polynomial(p_c_coeffs)
    return polynomial_to_euler_polynomial_e(p_c)
