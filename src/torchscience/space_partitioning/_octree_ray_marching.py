"""Octree ray marching operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._octree import Octree


def octree_ray_marching(
    tree: "Octree",
    origins: Tensor,
    directions: Tensor,
    *,
    step_size: float | None = None,
    maximum_steps: int = 512,
) -> tuple[Tensor, Tensor, Tensor]:
    """March rays through the octree using hierarchical DDA traversal.

    Parameters
    ----------
    tree : Octree
        Sparse voxel structure with full hierarchy.
    origins : Tensor, shape (n_rays, 3)
        Ray origins (any position, not restricted to [-1, 1]^3).
    directions : Tensor, shape (n_rays, 3)
        Ray directions (will be normalized).
    step_size : float, optional
        Fixed step size. If None, uses hierarchical voxel-adaptive stepping.
    maximum_steps : int, default=512
        Maximum samples per ray.

    Returns
    -------
    positions : Tensor, shape (n_rays, maximum_steps, 3)
        Sample positions along each ray.
    data : Tensor, shape (n_rays, maximum_steps, *value_shape)
        Sampled voxel data at each position.
    mask : Tensor, shape (n_rays, maximum_steps)
        True for valid samples (hit occupied voxel), False for padding.

    Notes
    -----
    **Bounding volume handling:** Rays are intersected with the [-1, 1]^3 AABB:
    - Marching starts at ray entry point (t_near), stops at exit (t_far)
    - Origins outside the cube start marching at AABB entry
    - Rays that miss the AABB entirely return all-False mask
    - This is consistent across CPU and CUDA backends

    **Fixed-size output:** Returns fixed-size tensors with a mask indicating
    valid samples. This enables torch.compile compatibility.

    **Hierarchical DDA (iterative implementation):** Traversal uses an explicit
    stack (max size = maximum_depth) rather than recursion, for CUDA compatibility
    and bounded compilation:

    1. Intersect ray with [-1, 1]^3 AABB to get (t_near, t_far)
    2. Start at root (depth 0), push to stack
    3. Pop voxel from stack, use DDA for ray-voxel intersection
    4. If voxel has children (children_mask != 0), push intersected children
    5. If voxel is leaf (children_mask == 0), sample data
    6. Step sizes adapt to voxel size: 2.0 / 2^depth
    7. Repeat until stack empty or maximum_steps reached

    **Autograd (qualified):**
    - With ``step_size`` provided (fixed stepping): Full gradients w.r.t.
      ``origins``, ``directions``, and ``tree.data``. Positions are differentiable
      via ``position = origin + t * direction``.
    - With ``step_size=None`` (adaptive stepping): Gradients w.r.t. ``tree.data``
      only. Position gradients are zeroed (detached) since hierarchical DDA
      traversal decisions are discontinuous.
    - **Boolean outputs (``mask``) have NO gradients.** Backward returns None/zeros
      for ``grad_mask``.

    Examples
    --------
    >>> tree = octree(points, features, maximum_depth=8)
    >>> origins = torch.tensor([[-1.0, 0.0, 0.0]])
    >>> directions = torch.tensor([[1.0, 0.0, 0.0]])
    >>> positions, data, mask = octree_ray_marching(tree, origins, directions)
    >>> valid_data = data[mask]  # Extract only valid samples
    """
    # Get maximum_depth as int
    max_depth = tree.maximum_depth.item()

    # Call C++ kernel
    return torch.ops.torchscience.octree_ray_marching(
        tree.data,
        tree.codes,
        tree.structure,
        tree.children_mask,
        origins,
        directions,
        max_depth,
        step_size,
        maximum_steps,
    )
