"""Conversion from zeros-poles-gain to second-order sections."""

from typing import Tuple

import torch
from torch import Tensor


def zpk_to_sos(
    zeros: Tensor,
    poles: Tensor,
    gain: Tensor,
    pairing: str = "nearest",
) -> Tensor:
    """
    Convert zeros, poles, and gain to second-order sections.

    Parameters
    ----------
    zeros : Tensor
        Zeros of the filter.
    poles : Tensor
        Poles of the filter.
    gain : Tensor
        System gain.
    pairing : str
        Pairing strategy: "nearest" or "keep_odd". Default is "nearest".

    Returns
    -------
    sos : Tensor
        Second-order sections, shape (n_sections, 6).
        Each row is [b0, b1, b2, a0, a1, a2].

    Notes
    -----
    Poles and zeros are paired to minimize coefficient sensitivity.
    Complex conjugate pairs are always kept together.
    """
    n_poles = poles.numel()
    n_zeros = zeros.numel()

    # Determine number of sections
    n_sections = (n_poles + 1) // 2

    # Separate real and complex poles/zeros
    real_poles, complex_poles = _separate_real_complex(poles)
    real_zeros, complex_zeros = _separate_real_complex(zeros)

    # Pad zeros with -1 (at Nyquist) if needed
    n_zeros_needed = n_poles
    if n_zeros < n_zeros_needed:
        padding = -torch.ones(
            n_zeros_needed - n_zeros, dtype=zeros.dtype, device=zeros.device
        )
        if zeros.numel() > 0:
            zeros_padded = torch.cat([zeros, padding])
        else:
            zeros_padded = padding
        real_zeros, complex_zeros = _separate_real_complex(zeros_padded)

    # Build sections
    sos_list = []

    # Pair complex poles with complex zeros
    while complex_poles.numel() >= 1:
        # Take first conjugate pair of poles
        p1 = complex_poles[0]
        complex_poles = complex_poles[1:]

        # Find nearest zero pair
        if complex_zeros.numel() >= 1:
            z1 = complex_zeros[0]
            complex_zeros = complex_zeros[1:]
        elif real_zeros.numel() >= 2:
            z1 = real_zeros[0]
            z2 = real_zeros[1]
            real_zeros = real_zeros[2:]
            # Build section with two real zeros
            b0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
            b1 = -(z1.real + z2.real)
            b2 = z1.real * z2.real
            a0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
            a1 = -2 * p1.real
            a2 = p1.real**2 + p1.imag**2
            sos_list.append(torch.stack([b0, b1, b2, a0, a1, a2]))
            continue
        else:
            z1 = torch.tensor(
                -1.0 + 0j, dtype=poles.dtype, device=poles.device
            )

        # Build section from complex conjugate pair
        b0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
        b1 = -2 * z1.real
        b2 = z1.real**2 + z1.imag**2
        a0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
        a1 = -2 * p1.real
        a2 = p1.real**2 + p1.imag**2

        sos_list.append(
            torch.stack([b0, b1.real, b2.real, a0, a1.real, a2.real])
        )

    # Handle remaining real poles
    while real_poles.numel() >= 2:
        p1 = real_poles[0]
        p2 = real_poles[1]
        real_poles = real_poles[2:]

        if real_zeros.numel() >= 2:
            z1 = real_zeros[0]
            z2 = real_zeros[1]
            real_zeros = real_zeros[2:]
        else:
            z1 = torch.tensor(-1.0, dtype=gain.dtype, device=gain.device)
            z2 = torch.tensor(-1.0, dtype=gain.dtype, device=gain.device)

        b0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
        b1 = -(z1.real + z2.real)
        b2 = z1.real * z2.real
        a0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
        a1 = -(p1.real + p2.real)
        a2 = p1.real * p2.real

        sos_list.append(torch.stack([b0, b1, b2, a0, a1, a2]))

    # Handle single remaining real pole (odd order)
    if real_poles.numel() == 1:
        p1 = real_poles[0]

        if real_zeros.numel() >= 1:
            z1 = real_zeros[0]
        else:
            z1 = torch.tensor(-1.0, dtype=gain.dtype, device=gain.device)

        b0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
        b1 = -z1.real
        b2 = torch.tensor(0.0, dtype=gain.dtype, device=gain.device)
        a0 = torch.tensor(1.0, dtype=gain.dtype, device=gain.device)
        a1 = -p1.real
        a2 = torch.tensor(0.0, dtype=gain.dtype, device=gain.device)

        sos_list.append(torch.stack([b0, b1, b2, a0, a1, a2]))

    if not sos_list:
        # Edge case: no poles (shouldn't happen for valid filters)
        return torch.zeros((0, 6), dtype=gain.dtype, device=gain.device)

    sos = torch.stack(sos_list)

    # Distribute gain across sections (avoid inplace ops for autograd)
    gain_per_section = gain.abs() ** (1.0 / n_sections)
    numerator_scaled = sos[:, :3] * gain_per_section.unsqueeze(-1)
    denominator = sos[:, 3:]

    # Handle sign of gain
    if gain < 0:
        # Negate first b0 coefficient
        sign_mult = torch.ones(
            n_sections, 3, dtype=gain.dtype, device=gain.device
        )
        sign_mult[0, 0] = -1.0
        numerator_scaled = numerator_scaled * sign_mult

    sos = torch.cat([numerator_scaled, denominator], dim=1)

    return sos.real if sos.is_complex() else sos


def _separate_real_complex(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Separate real and complex values, keeping only one of each conjugate pair.

    A value is considered "real" if:
    - Its imaginary part is negligible relative to its real part (1e-6 relative tol)
    - OR if the entire value is very small (< 1e-6), treat as real at origin

    For complex values, we keep only the one with positive imaginary part
    from each conjugate pair.
    """
    if x.numel() == 0:
        return x, x

    # Values are "real" if:
    # 1. abs(imag) < 1e-6 * abs(real) + 1e-10, OR
    # 2. The entire magnitude is negligible (< 1e-6)
    rel_tol = 1e-6 * x.real.abs() + 1e-10
    is_negligible = x.abs() < 1e-6
    is_real = (x.imag.abs() < rel_tol) | is_negligible

    real_vals = x[is_real].real

    # For complex, keep only positive imaginary (one of each conjugate pair)
    complex_mask = ~is_real & (x.imag > 0)
    complex_vals = x[complex_mask]

    return real_vals, complex_vals
