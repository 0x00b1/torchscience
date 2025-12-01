"""Closest point queries using BVH."""

from __future__ import annotations

import torch
from tensordict import tensorclass
from torch import Tensor

from torchscience.space_partitioning import BoundingVolumeHierarchy


@tensorclass
class ClosestPoint:
    """Closest point query results.

    Attributes
    ----------
    point : Tensor
        Closest point on surface, shape (*, 3).
    distance : Tensor
        Distance to closest point, shape (*,).
    geometry_id : Tensor
        Which mesh, shape (*,).
    primitive_id : Tensor
        Which triangle (global index), shape (*,).
    u : Tensor
        Barycentric u coordinate, shape (*,).
    v : Tensor
        Barycentric v coordinate, shape (*,).
    """

    point: Tensor
    distance: Tensor
    geometry_id: Tensor
    primitive_id: Tensor
    u: Tensor
    v: Tensor


class _ClosestPointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh_handle, query_points):
        point, distance, geometry_id, primitive_id, u, v = (
            torch.ops.torchscience.bvh_closest_point(
                bvh_handle,
                query_points.contiguous(),
            )
        )
        ctx.save_for_backward(query_points, point, distance)
        return point, distance, geometry_id, primitive_id, u, v

    @staticmethod
    def backward(
        ctx, grad_point, grad_distance, grad_gid, grad_pid, grad_u, grad_v
    ):
        query_points, closest_points, distance = ctx.saved_tensors

        grad_query = None

        if ctx.needs_input_grad[1]:  # query_points
            # closest_point is the projection of query onto surface
            # d(closest_point)/d(query) depends on local surface geometry

            hit_mask = torch.isfinite(distance)

            # d(distance)/d(query) = (query - closest) / distance (unit direction)
            direction = query_points - closest_points
            dist_safe = distance.unsqueeze(-1).clamp(min=1e-8)
            unit_dir = direction / dist_safe

            # For point output: d(point)/d(query) = I - n⊗n where n is surface normal
            # The surface normal at the closest point is the unit direction from closest to query
            # So d(point)/d(query) = I - unit_dir⊗unit_dir (projection onto tangent plane)
            # grad_from_point = grad_point - (grad_point · unit_dir) * unit_dir
            dot_product = (grad_point * unit_dir).sum(dim=-1, keepdim=True)
            grad_from_point = grad_point - dot_product * unit_dir

            # For distance output: d(dist)/d(query) = unit_direction
            grad_from_distance = grad_distance.unsqueeze(-1) * unit_dir

            grad_query = grad_from_point + grad_from_distance
            grad_query = grad_query * hit_mask.unsqueeze(-1).float()

        return None, grad_query


def closest_point(
    bvh: BoundingVolumeHierarchy,
    query_points: Tensor,
) -> ClosestPoint:
    """Find closest points on mesh surface.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        BVH structure to query.
    query_points : Tensor, shape (..., 3)
        Points to find closest surface points for.

    Returns
    -------
    ClosestPoint
        Query results with shape (...,).

    Notes
    -----
    This implementation uses 6-direction ray casting (along the positive and
    negative x, y, and z axes) to find the closest point. This is an
    approximation that may miss the actual closest point in some cases,
    particularly for complex geometry where the closest surface point is not
    visible from any of the six cardinal directions.
    """
    if query_points.shape[-1] != 3:
        raise ValueError(
            f"query_points must have shape (..., 3), got {query_points.shape}"
        )

    scene_handle = bvh._scene_handles.item()

    point, distance, geometry_id, primitive_id, u, v = (
        _ClosestPointFunction.apply(scene_handle, query_points)
    )

    return ClosestPoint(
        point=point,
        distance=distance,
        geometry_id=geometry_id,
        primitive_id=primitive_id,
        u=u,
        v=v,
        batch_size=list(query_points.shape[:-1]),
    )
