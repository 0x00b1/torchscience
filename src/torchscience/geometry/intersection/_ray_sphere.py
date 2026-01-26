"""Ray-sphere intersection."""

from __future__ import annotations

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.geometry.intersection._ray_hit import IntersectionResult


def ray_sphere(
    origins: Tensor,
    directions: Tensor,
    centers: Tensor,
    radii: Tensor,
) -> IntersectionResult:
    r"""Compute ray-sphere intersection.

    Finds the intersection point between rays and spheres defined
    by their centers and radii.

    Mathematical Definition
    -----------------------
    The sphere is defined by:

    .. math::
        \|\mathbf{x} - \mathbf{c}\|^2 = r^2

    where :math:`\mathbf{c}` is the sphere center and :math:`r` is the radius.

    The ray is:

    .. math::
        \mathbf{p}(t) = \mathbf{o} + t \cdot \mathbf{d}

    Substituting the ray equation into the sphere equation yields a quadratic:

    .. math::
        at^2 + bt + c = 0

    where:
    - :math:`a = \mathbf{d} \cdot \mathbf{d}`
    - :math:`b = 2(\mathbf{d} \cdot (\mathbf{o} - \mathbf{c}))`
    - :math:`c = \|\mathbf{o} - \mathbf{c}\|^2 - r^2`

    The smallest positive :math:`t` from the quadratic formula gives
    the nearest intersection.

    Parameters
    ----------
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors (need not be normalized).
    centers : Tensor, shape (..., 3)
        Sphere center points.
    radii : Tensor, shape (...,)
        Sphere radii (must be positive).

    Returns
    -------
    IntersectionResult
        Intersection results with fields:
        - t: Hit distance (inf if miss)
        - hit_point: World-space intersection point
        - normal: Surface normal at hit (points outward)
        - uv: Spherical coordinates (u=phi/2pi, v=theta/pi)
        - hit: Boolean mask for valid hits

    Examples
    --------
    Ray hitting a unit sphere at the origin:

    >>> import torch
    >>> from torchscience.geometry.intersection import ray_sphere
    >>> origins = torch.tensor([[0.0, 0.0, -5.0]])
    >>> directions = torch.tensor([[0.0, 0.0, 1.0]])
    >>> centers = torch.tensor([[0.0, 0.0, 0.0]])
    >>> radii = torch.tensor([1.0])
    >>> hit = ray_sphere(origins, directions, centers, radii)
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
    if centers.shape[-1] != 3:
        raise ValueError(
            f"centers must have shape (..., 3), got {centers.shape}"
        )

    # Ensure consistent dtype and device
    device = origins.device
    dtype = origins.dtype

    directions = directions.to(device=device, dtype=dtype)
    centers = centers.to(device=device, dtype=dtype)
    radii = radii.to(device=device, dtype=dtype)

    t, hit_point, normal, uv, hit = torch.ops.torchscience.ray_sphere(
        origins, directions, centers, radii
    )

    return IntersectionResult(
        t=t,
        hit_point=hit_point,
        normal=normal,
        uv=uv,
        hit=hit,
        batch_size=list(t.shape),
    )
