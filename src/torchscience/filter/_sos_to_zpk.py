"""Conversion from second-order sections to zeros-poles-gain."""

from typing import Tuple

import torch
from torch import Tensor

from ._exceptions import SOSNormalizationError


def sos_to_zpk(
    sos: Tensor,
    validate: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert second-order sections to zeros, poles, gain.

    Parameters
    ----------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].
    validate : bool, default True
        If True, validate SOS normalization (a0 = 1).

    Returns
    -------
    zeros : Tensor
        Zeros of the filter.
    poles : Tensor
        Poles of the filter.
    gain : Tensor
        System gain.

    Raises
    ------
    SOSNormalizationError
        If validate=True and a0 != 1 for any section.

    Notes
    -----
    Each second-order section represents a biquad filter:

    .. math::
        H_k(z) = \\frac{b_{k0} + b_{k1}z^{-1} + b_{k2}z^{-2}}{a_{k0} + a_{k1}z^{-1} + a_{k2}z^{-2}}

    The overall transfer function is:

    .. math::
        H(z) = \\prod_{k=0}^{n-1} H_k(z)

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import sos_to_zpk
    >>> sos = torch.tensor([[1.0, 2.0, 1.0, 1.0, -0.5, 0.1]])
    >>> zeros, poles, gain = sos_to_zpk(sos)
    """
    if sos.numel() == 0:
        dtype = sos.dtype if not sos.is_complex() else torch.complex128
        return (
            torch.zeros(0, dtype=torch.complex128, device=sos.device),
            torch.zeros(0, dtype=torch.complex128, device=sos.device),
            torch.ones(1, dtype=dtype, device=sos.device),
        )

    n_sections = sos.shape[0]

    # Validate normalization if requested
    if validate:
        a0_vals = sos[:, 3]
        if not torch.allclose(a0_vals, torch.ones_like(a0_vals), atol=1e-10):
            raise SOSNormalizationError(
                f"SOS sections must have a0 = 1, got a0 values: {a0_vals.tolist()}"
            )

    zeros_list = []
    poles_list = []
    gain = torch.ones(1, dtype=sos.dtype, device=sos.device)

    for i in range(n_sections):
        b0, b1, b2, a0, a1, a2 = sos[i]

        # Accumulate gain from b0 coefficients
        gain = gain * b0

        # Find zeros (roots of numerator: b0 + b1*z^-1 + b2*z^-2 = 0)
        # Multiply by z^2: b0*z^2 + b1*z + b2 = 0
        if b0.abs() > 1e-10:
            # Normalize to monic polynomial
            b1_norm = b1 / b0
            b2_norm = b2 / b0

            # Check if b2 is essentially zero (first-order section)
            if b2_norm.abs() < 1e-10:
                # First-order: b0*z + b1 = 0 -> z = -b1/b0
                if b1_norm.abs() > 1e-10:
                    z = -b1_norm
                    zeros_list.append(z.to(torch.complex128))
            else:
                # Second-order: z^2 + (b1/b0)*z + (b2/b0) = 0
                z1, z2 = _quadratic_roots(
                    torch.tensor(1.0, dtype=sos.dtype, device=sos.device),
                    b1_norm,
                    b2_norm,
                )
                zeros_list.append(z1)
                zeros_list.append(z2)

        # Find poles (roots of denominator: a0 + a1*z^-1 + a2*z^-2 = 0)
        # Multiply by z^2: a0*z^2 + a1*z + a2 = 0
        if a0.abs() > 1e-10:
            # Normalize to monic polynomial
            a1_norm = a1 / a0
            a2_norm = a2 / a0

            # Check if a2 is essentially zero (first-order section)
            if a2_norm.abs() < 1e-10:
                # First-order: a0*z + a1 = 0 -> z = -a1/a0
                if a1_norm.abs() > 1e-10:
                    p = -a1_norm
                    poles_list.append(p.to(torch.complex128))
            else:
                # Second-order: z^2 + (a1/a0)*z + (a2/a0) = 0
                p1, p2 = _quadratic_roots(
                    torch.tensor(1.0, dtype=sos.dtype, device=sos.device),
                    a1_norm,
                    a2_norm,
                )
                poles_list.append(p1)
                poles_list.append(p2)

    # Stack results
    if zeros_list:
        zeros = torch.stack(zeros_list)
    else:
        zeros = torch.zeros(0, dtype=torch.complex128, device=sos.device)

    if poles_list:
        poles = torch.stack(poles_list)
    else:
        poles = torch.zeros(0, dtype=torch.complex128, device=sos.device)

    return zeros, poles, gain


def _quadratic_roots(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute roots of quadratic equation ax^2 + bx + c = 0.

    Parameters
    ----------
    a, b, c : Tensor
        Coefficients of the quadratic.

    Returns
    -------
    r1, r2 : Tensor
        The two roots (may be complex).
    """
    # Convert to complex for potentially complex roots
    a = a.to(torch.complex128)
    b = b.to(torch.complex128)
    c = c.to(torch.complex128)

    discriminant = b * b - 4 * a * c
    sqrt_disc = torch.sqrt(discriminant)

    r1 = (-b + sqrt_disc) / (2 * a)
    r2 = (-b - sqrt_disc) / (2 * a)

    return r1, r2
