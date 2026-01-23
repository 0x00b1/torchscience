"""Subtract two Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def euler_polynomial_e_subtract(
    a: EulerPolynomialE,
    b: EulerPolynomialE,
) -> EulerPolynomialE:
    """Subtract two Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        First series.
    b : EulerPolynomialE
        Second series.

    Returns
    -------
    EulerPolynomialE
        Difference a - b.
    """
    a_coeffs = a.as_subclass(torch.Tensor)
    b_coeffs = b.as_subclass(torch.Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]
    n_max = max(n_a, n_b)

    # Pad shorter series with zeros
    if n_a < n_max:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_max - n_a
        a_coeffs = torch.cat(
            [
                a_coeffs,
                torch.zeros(
                    pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
                ),
            ],
            dim=-1,
        )

    if n_b < n_max:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_max - n_b
        b_coeffs = torch.cat(
            [
                b_coeffs,
                torch.zeros(
                    pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
                ),
            ],
            dim=-1,
        )

    return euler_polynomial_e(a_coeffs - b_coeffs)
