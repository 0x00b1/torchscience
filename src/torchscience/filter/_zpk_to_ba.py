"""Conversion from zeros-poles-gain to transfer function coefficients."""

from typing import Tuple

import torch
from torch import Tensor


def zpk_to_ba(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Convert zeros, poles, gain to transfer function coefficients.

    Parameters
    ----------
    zeros : Tensor
        Zeros of the transfer function.
    poles : Tensor
        Poles of the transfer function.
    gain : Tensor
        Gain of the transfer function.

    Returns
    -------
    numerator : Tensor
        Numerator polynomial coefficients in descending order.
    denominator : Tensor
        Denominator polynomial coefficients in descending order.

    Notes
    -----
    The transfer function is:

    .. math::
        H(z) = k \\frac{(z - z_0)(z - z_1)...}{(z - p_0)(z - p_1)...}

    which is converted to:

    .. math::
        H(z) = \\frac{b_0 + b_1 z^{-1} + ...}{a_0 + a_1 z^{-1} + ...}

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import zpk_to_ba
    >>> zeros = torch.tensor([-1.0 + 0j])
    >>> poles = torch.tensor([-0.5 + 0j])
    >>> gain = torch.tensor(0.25)
    >>> b, a = zpk_to_ba(zeros, poles, gain)
    """
    # Build polynomials from roots
    b = gain * _poly_from_roots(zeros)
    a = _poly_from_roots(poles)

    # Return real parts (imaginary should be negligible for conjugate pairs)
    return b.real, a.real


def _poly_from_roots(roots: Tensor) -> Tensor:
    """Build polynomial coefficients from roots.

    Computes coefficients of: (x - r0)(x - r1)...(x - rn)

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial.

    Returns
    -------
    coeffs : Tensor
        Polynomial coefficients in descending order [x^n, x^(n-1), ..., x^0].
    """
    if roots.numel() == 0:
        return torch.ones(1, dtype=roots.dtype, device=roots.device)

    # Start with (x - r0)
    coeffs = torch.tensor(
        [1.0, -roots[0]], dtype=torch.complex128, device=roots.device
    )

    # Multiply by each (x - ri)
    for r in roots[1:]:
        # Convolve [1, -r] with current coefficients
        new_coeffs = torch.zeros(
            len(coeffs) + 1, dtype=torch.complex128, device=roots.device
        )
        for i, c in enumerate(coeffs):
            new_coeffs[i] += c
            new_coeffs[i + 1] -= c * r
        coeffs = new_coeffs

    return coeffs
