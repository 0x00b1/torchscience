"""Add two Probabilists' Hermite series."""

from __future__ import annotations

import torch

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_add(
    a: HermitePolynomialHe,
    b: HermitePolynomialHe,
) -> HermitePolynomialHe:
    """Add two Probabilists' Hermite series.

    Parameters
    ----------
    a : HermitePolynomialHe
        First series.
    b : HermitePolynomialHe
        Second series.

    Returns
    -------
    HermitePolynomialHe
        Sum a + b.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.

    Examples
    --------
    >>> a = hermite_polynomial_he(torch.tensor([1.0, 2.0]))
    >>> b = hermite_polynomial_he(torch.tensor([3.0, 4.0, 5.0]))
    >>> c = hermite_polynomial_he_add(a, b)
    >>> c.coeffs
    tensor([4., 6., 5.])
    """
    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return HermitePolynomialHe(coeffs=a_coeffs + b_coeffs)

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(
            pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
        )
        a_coeffs = torch.cat([a_coeffs, padding], dim=-1)
    else:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(
            pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
        )
        b_coeffs = torch.cat([b_coeffs, padding], dim=-1)

    return HermitePolynomialHe(coeffs=a_coeffs + b_coeffs)
