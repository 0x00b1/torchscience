"""Ray-triangle intersection."""

from __future__ import annotations

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.geometry.intersection._ray_hit import IntersectionResult


def ray_triangle(
    origins: Tensor,
    directions: Tensor,
    v0: Tensor,
    v1: Tensor,
    v2: Tensor,
) -> IntersectionResult:
    r"""Compute ray-triangle intersection using the Moller-Trumbore algorithm.

    Finds the intersection point between rays and triangles defined
    by three vertices.

    Mathematical Definition
    -----------------------
    The triangle is defined by vertices :math:`\mathbf{v}_0`,
    :math:`\mathbf{v}_1`, and :math:`\mathbf{v}_2`. The ray is:

    .. math::
        \mathbf{p}(t) = \mathbf{o} + t \cdot \mathbf{d}

    The Moller-Trumbore algorithm solves for :math:`t`, :math:`u`, and
    :math:`v` in:

    .. math::
        \mathbf{o} + t \cdot \mathbf{d} = (1 - u - v) \mathbf{v}_0
        + u \mathbf{v}_1 + v \mathbf{v}_2

    by computing:

    .. math::
        \mathbf{e}_1 &= \mathbf{v}_1 - \mathbf{v}_0 \\
        \mathbf{e}_2 &= \mathbf{v}_2 - \mathbf{v}_0 \\
        \mathbf{h} &= \mathbf{d} \times \mathbf{e}_2 \\
        a &= \mathbf{e}_1 \cdot \mathbf{h}

    If :math:`|a| < \epsilon`, the ray is parallel to the triangle. Otherwise:

    .. math::
        f &= 1 / a \\
        \mathbf{s} &= \mathbf{o} - \mathbf{v}_0 \\
        u &= f \cdot (\mathbf{s} \cdot \mathbf{h}) \\
        \mathbf{q} &= \mathbf{s} \times \mathbf{e}_1 \\
        v &= f \cdot (\mathbf{d} \cdot \mathbf{q}) \\
        t &= f \cdot (\mathbf{e}_2 \cdot \mathbf{q})

    A valid hit requires :math:`u \geq 0`, :math:`v \geq 0`,
    :math:`u + v \leq 1`, and :math:`t > 0`.

    Parameters
    ----------
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors (need not be normalized).
    v0 : Tensor, shape (..., 3)
        First triangle vertex.
    v1 : Tensor, shape (..., 3)
        Second triangle vertex.
    v2 : Tensor, shape (..., 3)
        Third triangle vertex.

    Returns
    -------
    IntersectionResult
        Intersection results with fields:
        - t: Hit distance (inf if miss)
        - hit_point: World-space intersection point
        - normal: Surface normal at hit (cross product of edges)
        - uv: Barycentric coordinates (u, v)
        - hit: Boolean mask for valid hits

    Examples
    --------
    Ray hitting a triangle:

    >>> import torch
    >>> from torchscience.geometry.intersection import ray_triangle
    >>> origins = torch.tensor([[0.0, 0.0, -1.0]])
    >>> directions = torch.tensor([[0.0, 0.0, 1.0]])
    >>> v0 = torch.tensor([[-1.0, -1.0, 0.0]])
    >>> v1 = torch.tensor([[1.0, -1.0, 0.0]])
    >>> v2 = torch.tensor([[0.0, 1.0, 0.0]])
    >>> hit = ray_triangle(origins, directions, v0, v1, v2)
    >>> hit.t
    tensor([1.])
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
    if v0.shape[-1] != 3:
        raise ValueError(f"v0 must have shape (..., 3), got {v0.shape}")
    if v1.shape[-1] != 3:
        raise ValueError(f"v1 must have shape (..., 3), got {v1.shape}")
    if v2.shape[-1] != 3:
        raise ValueError(f"v2 must have shape (..., 3), got {v2.shape}")

    # Ensure consistent dtype and device
    device = origins.device
    dtype = origins.dtype

    directions = directions.to(device=device, dtype=dtype)
    v0 = v0.to(device=device, dtype=dtype)
    v1 = v1.to(device=device, dtype=dtype)
    v2 = v2.to(device=device, dtype=dtype)

    t, hit_point, normal, uv, hit = torch.ops.torchscience.ray_triangle(
        origins, directions, v0, v1, v2
    )

    return IntersectionResult(
        t=t,
        hit_point=hit_point,
        normal=normal,
        uv=uv,
        hit=hit,
        batch_size=list(t.shape),
    )
