"""Spotlight light source implementation."""

from typing import Tuple, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def spotlight(
    light_pos: Tensor,
    surface_pos: Tensor,
    spot_direction: Tensor,
    *,
    intensity: Union[Tensor, float],
    inner_angle: Union[Tensor, float],
    outer_angle: Union[Tensor, float],
) -> Tuple[Tensor, Tensor]:
    r"""Compute spotlight irradiance and direction.

    Evaluates spotlight illumination with angular falloff using smoothstep
    interpolation between inner and outer cone angles.

    Mathematical Definition
    -----------------------
    The angular falloff is computed as:

    .. math::
        \theta = \arccos(\mathbf{-L} \cdot \mathbf{d})

    where :math:`\mathbf{L}` is the normalized direction from surface to light
    and :math:`\mathbf{d}` is the spotlight direction.

    The falloff uses smoothstep interpolation:

    .. math::
        \text{falloff} = \text{smoothstep}(\cos(\theta_{outer}), \cos(\theta_{inner}), \cos(\theta))

    The irradiance is:

    .. math::
        E = \frac{I \cdot \text{falloff}}{d^2}

    Parameters
    ----------
    light_pos : Tensor, shape (..., 3)
        Position of the spotlight in world coordinates.
    surface_pos : Tensor, shape (..., 3)
        Position of the surface point to illuminate.
    spot_direction : Tensor, shape (..., 3)
        Direction the spotlight is pointing. Must be normalized.
    intensity : Tensor or float, shape (...) or scalar
        Light intensity (luminous intensity in candelas).
    inner_angle : Tensor or float, shape (...) or scalar
        Inner cone angle in radians. Full intensity inside this angle.
    outer_angle : Tensor or float, shape (...) or scalar
        Outer cone angle in radians. Zero intensity outside this angle.

    Returns
    -------
    irradiance : Tensor, shape (...)
        Irradiance at the surface point.
    light_dir : Tensor, shape (..., 3)
        Normalized direction from surface to light.

    Examples
    --------
    >>> light_pos = torch.tensor([[0.0, 5.0, 0.0]])
    >>> surface_pos = torch.tensor([[0.0, 0.0, 0.0]])
    >>> spot_direction = torch.tensor([[0.0, -1.0, 0.0]])
    >>> irradiance, light_dir = spotlight(
    ...     light_pos, surface_pos, spot_direction,
    ...     intensity=100.0, inner_angle=0.5, outer_angle=0.8
    ... )

    Notes
    -----
    - The spotlight direction should be normalized.
    - Returns zero irradiance for points outside the outer cone.
    - Uses inverse-square distance falloff.
    """
    if light_pos.shape[-1] != 3:
        raise ValueError(
            f"light_pos must have last dimension 3, got {light_pos.shape[-1]}"
        )
    if surface_pos.shape[-1] != 3:
        raise ValueError(
            f"surface_pos must have last dimension 3, got {surface_pos.shape[-1]}"
        )
    if spot_direction.shape[-1] != 3:
        raise ValueError(
            f"spot_direction must have last dimension 3, got {spot_direction.shape[-1]}"
        )

    if not isinstance(intensity, Tensor):
        intensity = torch.tensor(
            intensity, device=light_pos.device, dtype=light_pos.dtype
        )
    if not isinstance(inner_angle, Tensor):
        inner_angle = torch.tensor(
            inner_angle, device=light_pos.device, dtype=light_pos.dtype
        )
    if not isinstance(outer_angle, Tensor):
        outer_angle = torch.tensor(
            outer_angle, device=light_pos.device, dtype=light_pos.dtype
        )

    return torch.ops.torchscience.spotlight(
        light_pos,
        surface_pos,
        spot_direction,
        intensity,
        inner_angle,
        outer_angle,
    )
