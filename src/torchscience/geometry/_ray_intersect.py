"""Ray-triangle intersection using BVH."""

from __future__ import annotations

import torch
from tensordict import tensorclass
from torch import Tensor

from torchscience.space_partitioning import BoundingVolumeHierarchy


@tensorclass
class RayHit:
    """Ray intersection results."""

    t: Tensor
    hit: Tensor
    geometry_id: Tensor
    primitive_id: Tensor
    u: Tensor
    v: Tensor


class _RayIntersectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh_handle, origins, directions):
        t, hit, geometry_id, primitive_id, u, v = (
            torch.ops.torchscience.bvh_ray_intersect(
                bvh_handle,
                origins.contiguous(),
                directions.contiguous(),
            )
        )
        ctx.save_for_backward(t, hit, directions)
        return t, hit, geometry_id, primitive_id, u, v

    @staticmethod
    def backward(ctx, grad_t, grad_hit, grad_gid, grad_pid, grad_u, grad_v):
        t, hit, directions = ctx.saved_tensors

        # hit_point = origin + t * direction
        # d(hit_point)/d(origin) = I
        # d(hit_point)/d(direction) = t * I
        # But we're computing d(t)/d(origin) and d(t)/d(direction)

        # For t = (hit_point - origin) . n / (direction . n)
        # where n is triangle normal, this gets complex.
        # Simplified: gradient flows through t to origin/direction

        grad_origins = None
        grad_directions = None

        if ctx.needs_input_grad[1]:  # origins
            # d(t)/d(origin) ≈ -1/|direction| for rays pointing at triangle
            grad_origins = grad_t.unsqueeze(-1) * (
                -directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
            )
            grad_origins = grad_origins * hit.unsqueeze(-1).float()

        if ctx.needs_input_grad[2]:  # directions
            # d(t)/d(direction) ≈ -t * direction / |direction|^2
            dir_norm_sq = (directions * directions).sum(
                dim=-1, keepdim=True
            ) + 1e-8
            grad_directions = grad_t.unsqueeze(-1) * (
                -t.unsqueeze(-1) * directions / dir_norm_sq
            )
            grad_directions = grad_directions * hit.unsqueeze(-1).float()

        return None, grad_origins, grad_directions


def ray_intersect(
    bvh: BoundingVolumeHierarchy,
    origins: Tensor,
    directions: Tensor,
) -> RayHit:
    """Find ray-triangle intersections.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        BVH structure to query.
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors (need not be normalized).

    Returns
    -------
    RayHit
        Intersection results with shape (...,).
    """
    if origins.shape[-1] != 3:
        raise ValueError(
            f"origins must have shape (..., 3), got {origins.shape}"
        )
    if directions.shape[-1] != 3:
        raise ValueError(
            f"directions must have shape (..., 3), got {directions.shape}"
        )

    origins, directions = torch.broadcast_tensors(origins, directions)
    scene_handle = bvh._scene_handles.item()

    t, hit, geometry_id, primitive_id, u, v = _RayIntersectFunction.apply(
        scene_handle, origins, directions
    )

    return RayHit(
        t=t,
        hit=hit,
        geometry_id=geometry_id,
        primitive_id=primitive_id,
        u=u,
        v=v,
        batch_size=list(origins.shape[:-1]),
    )
