"""Ray-AABB intersection."""

from __future__ import annotations

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.geometry.intersection._ray_hit import IntersectionResult


def ray_aabb(
    origins: Tensor,
    directions: Tensor,
    box_min: Tensor,
    box_max: Tensor,
) -> IntersectionResult:
    r"""Compute ray-AABB intersection using the slab method.

    Finds the intersection point between rays and axis-aligned bounding
    boxes (AABBs) defined by their minimum and maximum corners.

    Mathematical Definition
    -----------------------
    An AABB is defined by two corners:

    .. math::
        \mathbf{b}_{\min} = (x_{\min}, y_{\min}, z_{\min}), \quad
        \mathbf{b}_{\max} = (x_{\max}, y_{\max}, z_{\max})

    The ray is:

    .. math::
        \mathbf{p}(t) = \mathbf{o} + t \cdot \mathbf{d}

    The slab method intersects the ray with each pair of axis-aligned
    planes (slabs). For each axis :math:`i`:

    .. math::
        t_{1,i} = \frac{b_{\min,i} - o_i}{d_i}, \quad
        t_{2,i} = \frac{b_{\max,i} - o_i}{d_i}

    The entry and exit parameters are:

    .. math::
        t_{\text{near}} = \max_i \min(t_{1,i}, t_{2,i}), \quad
        t_{\text{far}} = \min_i \max(t_{1,i}, t_{2,i})

    A hit occurs when :math:`t_{\text{near}} \leq t_{\text{far}}` and
    :math:`t_{\text{far}} \geq 0`.

    Parameters
    ----------
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors (need not be normalized).
    box_min : Tensor, shape (..., 3)
        AABB minimum corner points.
    box_max : Tensor, shape (..., 3)
        AABB maximum corner points.

    Returns
    -------
    IntersectionResult
        Intersection results with fields:
        - t: Hit distance (inf if miss)
        - hit_point: World-space intersection point
        - normal: Axis-aligned face normal at hit (points outward)
        - uv: Parametric coordinates on the hit face, mapped to [0, 1]
        - hit: Boolean mask for valid hits

    Examples
    --------
    Ray hitting a unit cube centered at the origin:

    >>> import torch
    >>> from torchscience.geometry.intersection import ray_aabb
    >>> origins = torch.tensor([[0.0, 0.0, -5.0]])
    >>> directions = torch.tensor([[0.0, 0.0, 1.0]])
    >>> box_min = torch.tensor([[-1.0, -1.0, -1.0]])
    >>> box_max = torch.tensor([[1.0, 1.0, 1.0]])
    >>> hit = ray_aabb(origins, directions, box_min, box_max)
    >>> hit.t
    tensor([4.])
    >>> hit.hit
    tensor([True])
    """
    if origins.shape[-1] != 3:
        raise ValueError(
            f"origins must have shape (..., 3), got {origins.shape}"
        )
    if directions.shape[-1] != 3:
        raise ValueError(
            f"directions must have shape (..., 3), got {directions.shape}"
        )
    if box_min.shape[-1] != 3:
        raise ValueError(
            f"box_min must have shape (..., 3), got {box_min.shape}"
        )
    if box_max.shape[-1] != 3:
        raise ValueError(
            f"box_max must have shape (..., 3), got {box_max.shape}"
        )

    # Ensure consistent dtype and device
    device = origins.device
    dtype = origins.dtype

    directions = directions.to(device=device, dtype=dtype)
    box_min = box_min.to(device=device, dtype=dtype)
    box_max = box_max.to(device=device, dtype=dtype)

    t, hit_point, normal, uv, hit = torch.ops.torchscience.ray_aabb(
        origins, directions, box_min, box_max
    )

    return IntersectionResult(
        t=t,
        hit_point=hit_point,
        normal=normal,
        uv=uv,
        hit=hit,
        batch_size=list(t.shape),
    )
