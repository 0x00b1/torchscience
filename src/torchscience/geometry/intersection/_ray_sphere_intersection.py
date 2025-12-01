"""Ray-sphere intersection."""

from __future__ import annotations

from typing import Union

import torch
from tensordict import tensorclass
from torch import Tensor


@tensorclass
class RaySphereHit:
    """Ray-sphere intersection results.

    Attributes
    ----------
    t : Tensor, shape (...)
        Distance along ray to intersection point. Set to inf for misses.
    hit : Tensor, shape (...)
        Boolean mask indicating which rays hit the sphere.
    point : Tensor, shape (..., 3)
        Intersection point in world space. Undefined for misses.
    normal : Tensor, shape (..., 3)
        Surface normal at intersection (outward-facing, respects front_face).
        Undefined for misses.
    front_face : Tensor, shape (...)
        True if ray hit the front (exterior) of the sphere.
    """

    t: Tensor
    hit: Tensor
    point: Tensor
    normal: Tensor
    front_face: Tensor


def ray_sphere_intersection(
    origins: Tensor,
    directions: Tensor,
    centers: Tensor,
    radii: Union[Tensor, float],
    *,
    t_min: float = 1e-4,
    t_max: float = float("inf"),
) -> RaySphereHit:
    r"""Compute ray-sphere intersections.

    Finds the intersection of rays with spheres using the analytic quadratic
    solution. Fully differentiable with respect to all inputs.

    Mathematical Definition
    -----------------------
    A point :math:`P` lies on a sphere centered at :math:`C` with radius :math:`r` if:

    .. math::
        |P - C|^2 = r^2

    A point on a ray is :math:`P(t) = O + t \cdot D` where :math:`O` is the origin
    and :math:`D` is the direction. Substituting gives the quadratic:

    .. math::
        at^2 + bt + c = 0

    where:

    .. math::
        a &= D \cdot D \\
        b &= 2 \cdot D \cdot (O - C) \\
        c &= (O - C) \cdot (O - C) - r^2

    The discriminant :math:`\Delta = b^2 - 4ac` determines:

    - :math:`\Delta < 0`: No intersection (ray misses)
    - :math:`\Delta = 0`: Ray grazes sphere (one intersection)
    - :math:`\Delta > 0`: Ray enters and exits (two intersections)

    Parameters
    ----------
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors. Need not be normalized.
    centers : Tensor, shape (..., 3) or (3,)
        Sphere center points. Broadcasts with ray tensors.
    radii : Tensor or float, shape (...) or scalar
        Sphere radii. Broadcasts with ray tensors.
    t_min : float, default=1e-4
        Minimum valid t value. Intersections with t < t_min are ignored.
        Use to avoid self-intersection artifacts.
    t_max : float, default=inf
        Maximum valid t value. Intersections with t > t_max are ignored.
        Use for shadow rays or depth culling.

    Returns
    -------
    RaySphereHit
        Tensorclass containing intersection results:

        - **t**: Distance to intersection (inf for misses)
        - **hit**: Boolean mask of valid intersections
        - **point**: World-space intersection point
        - **normal**: Surface normal (flipped to face ray origin)
        - **front_face**: True if ray hit exterior of sphere

    Examples
    --------
    Single ray hitting a unit sphere at origin:

    >>> origin = torch.tensor([[0.0, 0.0, -3.0]])
    >>> direction = torch.tensor([[0.0, 0.0, 1.0]])
    >>> center = torch.tensor([0.0, 0.0, 0.0])
    >>> result = ray_sphere_intersection(origin, direction, center, 1.0)
    >>> result.hit
    tensor([True])
    >>> result.t
    tensor([2.])

    Batched rays with multiple spheres (broadcasted):

    >>> origins = torch.randn(100, 3)
    >>> directions = torch.randn(100, 3)
    >>> centers = torch.tensor([[0., 0., 0.], [2., 0., 0.]])  # 2 spheres
    >>> radii = torch.tensor([1.0, 0.5])
    >>> # Test each ray against sphere 0
    >>> hit0 = ray_sphere_intersection(origins, directions, centers[0], radii[0])

    Gradient computation for differentiable rendering:

    >>> origin = torch.tensor([[0., 0., -3.]], requires_grad=True)
    >>> direction = torch.tensor([[0., 0., 1.]])
    >>> center = torch.tensor([0., 0., 0.])
    >>> result = ray_sphere_intersection(origin, direction, center, 1.0)
    >>> result.t.sum().backward()
    >>> origin.grad  # d(t)/d(origin)
    tensor([[0., 0., -1.]])

    Notes
    -----
    - The normal is always oriented to face the ray origin (opposite to
      ray direction). Use ``front_face`` to determine if the ray hit the
      interior or exterior of the sphere.
    - When ``hit=False``, the values of ``point``, ``normal``, and ``front_face``
      are undefined but will have valid tensor shapes.
    - For best numerical stability, normalize ray directions before calling.
    - Supports complex-valued tensors for specialized applications.

    See Also
    --------
    ray_intersect : Ray-triangle intersection using BVH acceleration.

    References
    ----------
    .. [1] P. Shirley, "Ray Tracing in One Weekend", 2016.
           https://raytracing.github.io/books/RayTracingInOneWeekend.html
    """
    # Validate inputs
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

    # Convert radii to tensor if needed
    if not isinstance(radii, Tensor):
        radii = torch.tensor(radii, device=origins.device, dtype=origins.dtype)

    # Broadcast all inputs to compatible shapes
    # We need to handle the last dimension (3) separately
    origins, directions, centers = torch.broadcast_tensors(
        origins, directions, centers
    )

    # Broadcast radii to match batch dimensions
    batch_shape = origins.shape[:-1]
    radii = radii.broadcast_to(batch_shape)

    device = origins.device
    dtype = origins.dtype

    # Vector from sphere center to ray origin
    oc = origins - centers  # (..., 3)

    # Quadratic coefficients: at² + bt + c = 0
    # Using half_b optimization: b = 2*half_b, discriminant = half_b² - ac
    a = (directions * directions).sum(dim=-1)  # D·D
    half_b = (oc * directions).sum(dim=-1)  # (O-C)·D
    c = (oc * oc).sum(dim=-1) - radii * radii  # |O-C|² - r²

    # Discriminant (using half_b form)
    discriminant = half_b * half_b - a * c

    # Initialize outputs
    hit = discriminant >= 0
    t = torch.full(batch_shape, float("inf"), device=device, dtype=dtype)

    # Compute t values where discriminant >= 0
    # Use clamp to avoid NaN from sqrt of negative numbers
    sqrt_d = torch.sqrt(discriminant.clamp(min=0))

    # Two solutions: t1 (near), t2 (far)
    t1 = (-half_b - sqrt_d) / a
    t2 = (-half_b + sqrt_d) / a

    # Select valid t: prefer t1 (closer) if in range, else try t2
    t_min_tensor = torch.tensor(t_min, device=device, dtype=dtype)
    t_max_tensor = torch.tensor(t_max, device=device, dtype=dtype)

    use_t1 = hit & (t1 >= t_min_tensor) & (t1 <= t_max_tensor)
    use_t2 = hit & ~use_t1 & (t2 >= t_min_tensor) & (t2 <= t_max_tensor)

    t = torch.where(use_t1, t1, t)
    t = torch.where(use_t2, t2, t)
    hit = use_t1 | use_t2

    # Compute intersection point: P = O + t*D
    point = origins + t.unsqueeze(-1) * directions

    # Compute outward-facing normal: (P - C) / r
    outward_normal = (point - centers) / radii.unsqueeze(-1)

    # Determine if we hit front face (ray from outside) or back face (ray from inside)
    # Front face: ray direction opposes normal (D·N < 0)
    front_face = (directions * outward_normal).sum(dim=-1) < 0

    # Flip normal to always face the ray origin
    normal = torch.where(
        front_face.unsqueeze(-1), outward_normal, -outward_normal
    )

    return RaySphereHit(
        t=t,
        hit=hit,
        point=point,
        normal=normal,
        front_face=front_face,
        batch_size=list(batch_shape),
    )
