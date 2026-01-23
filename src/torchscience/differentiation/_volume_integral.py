"""Volume integral computation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.amp import custom_fwd


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def volume_integral(
    field: Tensor,
    spacing: Tensor | Sequence[float] | float = 1.0,
    *,
    region: Tensor | None = None,
) -> Tensor:
    r"""Compute volume integral over a region.

    Computes the volume integral of a scalar field f over a region V:

    .. math::
        \iiint_V f \, dV

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    spacing : float or sequence of floats
        Grid spacing. Scalar for uniform, or sequence for anisotropic.
    region : Tensor, optional
        Boolean mask defining the integration region.
        If None, integrates over the entire domain.

    Returns
    -------
    Tensor
        Scalar value of the volume integral.

    Notes
    -----
    The volume integral is computed using the midpoint rule:

    .. math::
        \int f \, dV \approx \sum_i f_i \cdot dV

    For 2D, this computes an area integral.
    For 1D, this computes a line integral.

    Examples
    --------
    >>> # Total energy in a domain
    >>> energy_density = torch.randn(32, 32, 32)
    >>> dx = 0.01  # 1 cm grid spacing
    >>> total_energy = volume_integral(energy_density, spacing=dx)
    """
    ndim = field.ndim

    # Compute volume element dV
    if isinstance(spacing, (int, float)):
        dV = spacing**ndim
    elif isinstance(spacing, Tensor):
        dV = spacing.prod().item()
    else:
        dV = 1.0
        for s in spacing:
            dV *= s

    # Apply region mask if provided
    if region is not None:
        field = field * region.to(field.dtype)

    # Integrate
    result = field.sum() * dV

    return result
