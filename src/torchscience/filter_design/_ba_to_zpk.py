"""Conversion from transfer function coefficients to zeros-poles-gain."""

from typing import Tuple

import torch
from torch import Tensor


def ba_to_zpk(
    numerator: Tensor,
    denominator: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert transfer function coefficients to zeros, poles, gain.

    Parameters
    ----------
    numerator : Tensor
        Numerator polynomial coefficients in descending order.
    denominator : Tensor
        Denominator polynomial coefficients in descending order.

    Returns
    -------
    zeros : Tensor
        Zeros of the transfer function.
    poles : Tensor
        Poles of the transfer function.
    gain : Tensor
        Gain of the transfer function.

    Notes
    -----
    The transfer function:

    .. math::
        H(z) = \\frac{b_0 + b_1 z^{-1} + ... + b_n z^{-n}}{a_0 + a_1 z^{-1} + ... + a_m z^{-m}}

    is converted to:

    .. math::
        H(z) = k \\frac{(z - z_0)(z - z_1)...}{(z - p_0)(z - p_1)...}

    where k is the gain.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import ba_to_zpk
    >>> b = torch.tensor([0.25, 0.25])
    >>> a = torch.tensor([1.0, -0.5])
    >>> zeros, poles, gain = ba_to_zpk(b, a)
    """
    # Ensure complex dtype for root finding
    b = numerator.to(torch.complex128)
    a = denominator.to(torch.complex128)

    # Remove leading zeros from numerator and denominator
    b = _strip_leading_zeros(b)
    a = _strip_leading_zeros(a)

    # Find zeros (roots of numerator)
    if b.numel() <= 1:
        zeros = torch.zeros(0, dtype=torch.complex128, device=numerator.device)
    else:
        zeros = _polynomial_roots(b)

    # Find poles (roots of denominator)
    if a.numel() <= 1:
        poles = torch.zeros(
            0, dtype=torch.complex128, device=denominator.device
        )
    else:
        poles = _polynomial_roots(a)

    # Gain is ratio of leading coefficients
    gain = b[0] / a[0]

    return zeros, poles, gain.real.to(numerator.dtype)


def _strip_leading_zeros(coeffs: Tensor) -> Tensor:
    """Remove leading near-zero coefficients from polynomial."""
    if coeffs.numel() == 0:
        return coeffs

    # Find first non-zero coefficient
    nonzero_mask = coeffs.abs() > 1e-14
    if not nonzero_mask.any():
        return coeffs[-1:]  # Return last coefficient if all zeros

    first_nonzero = nonzero_mask.nonzero()[0, 0]
    return coeffs[first_nonzero:]


def _polynomial_roots(coeffs: Tensor) -> Tensor:
    """Find roots of polynomial using companion matrix method.

    Parameters
    ----------
    coeffs : Tensor
        Polynomial coefficients in descending order [c_n, c_{n-1}, ..., c_0]
        representing c_n*x^n + c_{n-1}*x^{n-1} + ... + c_0.

    Returns
    -------
    roots : Tensor
        Roots of the polynomial.
    """
    n = coeffs.numel() - 1
    if n <= 0:
        return torch.zeros(0, dtype=torch.complex128, device=coeffs.device)

    # Normalize to monic polynomial (leading coefficient = 1)
    coeffs = coeffs / coeffs[0]

    # Build companion matrix
    # For polynomial x^n + c_{n-1}*x^{n-1} + ... + c_0
    # Companion matrix has -coefficients in last column and 1s on subdiagonal
    companion = torch.zeros(
        (n, n), dtype=torch.complex128, device=coeffs.device
    )

    # Fill subdiagonal with 1s
    if n > 1:
        companion[1:, :-1] = torch.eye(
            n - 1, dtype=torch.complex128, device=coeffs.device
        )

    # Fill last column with negated coefficients (excluding leading 1)
    companion[:, -1] = -coeffs[1:]

    # Eigenvalues of companion matrix are roots of polynomial
    roots = torch.linalg.eigvals(companion)

    return roots
