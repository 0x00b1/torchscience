"""Trim trailing zeros from Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)


def euler_polynomial_e_trim(
    a: EulerPolynomialE,
    tol: float = 0.0,
) -> EulerPolynomialE:
    """Trim trailing near-zero coefficients from Euler polynomial series.

    Parameters
    ----------
    a : EulerPolynomialE
        Series to trim.
    tol : float, optional
        Tolerance for considering a coefficient as zero. Default is 0.0.

    Returns
    -------
    EulerPolynomialE
        Trimmed series.
    """
    coeffs = a.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Find last non-zero coefficient
    # For batched, consider all batches
    last_nonzero = 0
    for k in range(n):
        if torch.any(torch.abs(coeffs[..., k]) > tol):
            last_nonzero = k

    # Keep at least one coefficient
    new_n = max(last_nonzero + 1, 1)

    return euler_polynomial_e(coeffs[..., :new_n].clone())
