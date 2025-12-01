"""Ray occlusion queries using BVH."""

from __future__ import annotations

import torch
from torch import Tensor

from torchscience.space_partitioning import BoundingVolumeHierarchy


def ray_occluded(
    bvh: BoundingVolumeHierarchy,
    origins: Tensor,
    directions: Tensor,
) -> Tensor:
    """Test if rays are occluded by geometry.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        BVH structure to query.
    origins : Tensor, shape (..., 3)
        Ray origin points.
    directions : Tensor, shape (..., 3)
        Ray direction vectors.

    Returns
    -------
    Tensor
        Boolean tensor of shape (...) indicating occlusion.
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

    return torch.ops.torchscience.bvh_ray_occluded(
        scene_handle,
        origins.contiguous(),
        directions.contiguous(),
    )
