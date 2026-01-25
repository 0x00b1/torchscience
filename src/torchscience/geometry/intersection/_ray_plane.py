"""Ray-plane intersection."""

from __future__ import annotations

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.geometry.intersection._ray_hit import IntersectionResult


def ray_plane(
    origins: Tensor,
    directions: Tensor,
    plane_normals: Tensor,
    plane_offsets: Tensor,
) -> IntersectionResult:
    r"""Compute ray-plane intersection.

    Finds the intersection point between rays and infinite planes defined
    by their normal vectors and distances from origin.

    Mathematical Definition
    -----------------------
    The plane is defined by:

    .. math::
        \mathbf{n} \cdot \mathbf{x} = d

    where :math:`\mathbf{n}` is the plane normal and :math:`d` is the offset.

    The ray is:

    .. math::
        \mathbf{p}(t) = \mathbf{o} + t \cdot \mathbf{d}

    The intersection parameter is:

    .. math::
        t = \frac{d - \mathbf{n} \cdot \mathbf{o}}{\mathbf{n} \cdot \mathbf{d}}

    Parameters
    ----------
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors (need not be normalized).
    plane_normals : Tensor, shape (..., 3)
        Plane normal vectors.
    plane_offsets : Tensor, shape (...,)
        Plane offset from origin (signed distance along normal).

    Returns
    -------
    IntersectionResult
        Intersection results with fields:
        - t: Hit distance (inf if miss)
        - hit_point: World-space intersection point
        - normal: Surface normal at hit
        - uv: World-space (x, y) coordinates on plane
        - hit: Boolean mask for valid hits

    Examples
    --------
    Ray hitting ground plane:

    >>> import torch
    >>> from torchscience.geometry.intersection import ray_plane
    >>> origins = torch.tensor([[0.0, 5.0, 0.0]])
    >>> directions = torch.tensor([[0.0, -1.0, 0.0]])
    >>> plane_normals = torch.tensor([[0.0, 1.0, 0.0]])
    >>> plane_offsets = torch.tensor([0.0])
    >>> hit = ray_plane(origins, directions, plane_normals, plane_offsets)
    >>> hit.t
    tensor([5.])
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
    if plane_normals.shape[-1] != 3:
        raise ValueError(
            f"plane_normals must have shape (..., 3), got {plane_normals.shape}"
        )

    # Ensure consistent dtype and device
    device = origins.device
    dtype = origins.dtype

    directions = directions.to(device=device, dtype=dtype)
    plane_normals = plane_normals.to(device=device, dtype=dtype)
    plane_offsets = plane_offsets.to(device=device, dtype=dtype)

    t, hit_point, normal, uv, hit = torch.ops.torchscience.ray_plane(
        origins, directions, plane_normals, plane_offsets
    )

    return IntersectionResult(
        t=t,
        hit_point=hit_point,
        normal=normal,
        uv=uv,
        hit=hit,
        batch_size=list(t.shape),
    )
