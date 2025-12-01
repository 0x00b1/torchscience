"""Octree point query operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._octree import Octree


# Interpolation mode constants (must match C++ enum)
INTERPOLATION_NEAREST = 0
INTERPOLATION_TRILINEAR = 1


def octree_sample(
    tree: "Octree",
    points: Tensor,
    *,
    interpolation: str = "nearest",
    query_depth: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Query voxel data at given points.

    Parameters
    ----------
    tree : Octree
        Sparse voxel structure with full hierarchy.
    points : Tensor, shape (m, 3) or (*, m, 3)
        Query coordinates in [-1, 1]³. Supports batched queries.
    interpolation : str, default="nearest"
        "nearest" - Return data from deepest existing voxel containing point
        "trilinear" - Interpolate from 8 neighboring voxel centers using
                      top-down traversal per corner (missing corners contribute zero)
    query_depth : int, optional
        Maximum depth to query. If None, uses tree.maximum_depth.
        Query returns the deepest existing voxel at depth <= query_depth.

    Returns
    -------
    data : Tensor, shape (m, *value_shape) or (*, m, *value_shape)
        Retrieved voxel data. Zeros for points in empty regions.
    found : Tensor, shape (m,) or (*, m), dtype=bool
        True if query hit an occupied voxel, False for empty space.

    Notes
    -----
    **Top-down traversal semantics (CRITICAL):**

    Queries use top-down path traversal with ``children_mask`` to find the deepest
    voxel containing the query point. This correctly handles sparse regions:

    1. Let ``d_max = query_depth`` if provided, else ``tree.maximum_depth``
    2. Start at root (depth 0). If tree is empty, return ``found=False``
    3. For ``d = 0`` to ``d_max - 1``:
       a. If current node is a leaf (``children_mask == 0``), return it
          (handles merged coarse leaves)
       b. Compute octant index for query point at level ``d + 1``
       c. If ``children_mask`` does NOT include that octant, return ``found=False``
          (empty region - no voxel covers this point)
       d. Otherwise, look up child node by code and continue
    4. If we reach depth ``d_max``, return that node (leaf or internal)

    This gives correct behavior for:
    - **Empty regions:** Query in sparse area returns ``found=False`` (not ancestor)
    - **Coarse leaves:** Query inside merged leaf returns the coarse leaf's data
    - **Fine leaves:** Query reaches maximum depth and returns leaf data
    - **Coarse queries:** ``query_depth < maximum_depth`` returns aggregated internal data

    **Why NOT ancestor fallback:** A naive "walk up until found" approach would
    incorrectly return root data for any query in a non-empty tree, since the
    root always exists. The top-down approach uses ``children_mask`` to detect
    when the query point's path doesn't exist.

    **Multi-resolution queries:** Set ``query_depth`` to limit traversal depth.
    For example, ``query_depth=4`` stops at depth 4, returning internal node data
    (aggregated from descendants) if no coarse leaf is encountered earlier.

    **Autograd:** Supports gradients w.r.t. ``tree.data`` (both interpolation modes)
    and ``points`` (trilinear only).

    **Out-of-bounds handling:** Query points outside [-1, 1]³ are **silently
    clamped** to the boundary before lookup. This matches construction behavior
    and ensures out-of-bounds queries return boundary voxel data rather than
    spurious ``found=False`` results.

    Examples
    --------
    >>> tree = octree(points, features, maximum_depth=8)
    >>> queries = torch.rand(100, 3) * 2 - 1

    # Query at leaf level (default)
    >>> data, found = octree_sample(tree, queries)

    # Query at coarse level for fast approximate lookup
    >>> data_coarse, found_coarse = octree_sample(tree, queries, query_depth=4)

    # Trilinear interpolation at leaf level
    >>> data_smooth, found = octree_sample(
    ...     tree, queries, interpolation="trilinear"
    ... )
    """
    # Map interpolation string to integer enum
    interpolation_map = {
        "nearest": INTERPOLATION_NEAREST,
        "trilinear": INTERPOLATION_TRILINEAR,
    }
    if interpolation not in interpolation_map:
        raise RuntimeError(
            f"interpolation must be one of {list(interpolation_map.keys())}, "
            f"got '{interpolation}'"
        )
    interpolation_int = interpolation_map[interpolation]

    # Get maximum_depth as int
    max_depth = tree.maximum_depth.item()

    # Use provided query_depth or default to maximum_depth
    if query_depth is not None:
        if query_depth < 0 or query_depth > max_depth:
            raise RuntimeError(
                f"query_depth must be in [0, {max_depth}], got {query_depth}"
            )

    # Call C++ kernel
    return torch.ops.torchscience.octree_sample(
        tree.data,
        tree.codes,
        tree.structure,
        tree.children_mask,
        points,
        max_depth,
        interpolation_int,
        query_depth,
    )
