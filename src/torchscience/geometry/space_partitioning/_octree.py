"""Sparse voxel octree implementation with Morton-code hash table storage."""

from __future__ import annotations

import torch
from tensordict import tensorclass
from torch import Tensor

# Aggregation mode constants (must match C++ enum)
AGGREGATION_MEAN = 0
AGGREGATION_SUM = 1
AGGREGATION_MAX = 2


@tensorclass
class Octree:
    """Sparse voxel octree using Morton-code hash table with mixed-depth LOD.

    This class wraps the tree structure tensors returned by the C++ kernel.
    Use :func:`octree` to construct instances.

    As a tensorclass, Octree supports:
    - Device movement: ``tree.to("cuda")`` or ``tree.cuda()``
    - Serialization: ``torch.save(tree, path)`` / ``torch.load(path)``

    Attributes
    ----------
    codes : Tensor, shape (count,), dtype=int64
        Morton codes encoding (depth, x, y, z) for each voxel.
        Bits 60-63: depth level (0-15)
        Bits 0-59: interleaved x, y, z coordinates (20 bits each)
    data : Tensor, shape (count, *value_shape)
        Voxel data. Shape depends on use case:
        - (count, 1) for binary occupancy
        - (count, C) for feature vectors
        - (count, 3) for RGB or velocity fields
        For internal nodes, contains aggregated data from descendants.
    structure : Tensor, shape (capacity,), dtype=int64
        Hash table mapping hash(code) % capacity -> index in codes/data.
        Value of -1 indicates empty slot.
    children_mask : Tensor, shape (count,), dtype=uint8
        Bitmask indicating which child octants exist (bits 0-7).
        Value of 0 means leaf node (no children).
        Used for efficient hierarchical traversal.
    weights : Tensor, shape (count,), dtype=float32
        Number of points that contributed to each voxel's data.
        For leaves: count of input points mapped to this voxel.
        For internal nodes: sum of descendant weights (for weighted aggregation).
    maximum_depth : Tensor, scalar, dtype=int64
        Maximum octree depth. Resolution = 2^maximum_depth per axis.
    count : Tensor, scalar, dtype=int64
        Current number of voxels (leaves + internal nodes).
    aggregation : Tensor, scalar, dtype=int64
        Aggregation mode for internal node data and dynamic updates:
        0 = mean (weighted average), 1 = sum, 2 = max.

    Notes
    -----
    **Hierarchy invariant:** All ancestors of every leaf exist (complete path
    from root to each leaf). Internal nodes store aggregated data from their
    descendants. However, nodes *below* a merged leaf do not exist—merging
    removes finer children to save memory.

    **Leaf identification:** A voxel is a leaf if and only if
    ``children_mask == 0``. No separate is_leaf field is needed.

    **Mixed-depth LOD:** Leaves can exist at varying depths. After merge
    operations, coarse leaves (low depth) replace their finer children.
    After subdivision, fine leaves (high depth) are created from coarse parents.

    **Query semantics:** Point queries use top-down traversal with ``children_mask``
    to find the deepest existing voxel containing the query point. This correctly
    handles sparse regions (returning not-found) and mixed-depth leaves (returning
    coarse leaves that cover the query point).

    **Internal node data semantics:**

    Internal node data serves TWO distinct use cases with different semantics:

    1. **Cache mode (dynamic updates):** Internal node data is a derived cache
       computed via weighted aggregation from leaf descendants. Dynamic update
       operations recompute ancestor data automatically. In this mode,
       gradients to internal node data are NOT meaningful for training—only
       leaf data should be trained.

    2. **Independent parameterization (static tree):** Internal node data is
       treated as independent trainable parameters (multi-resolution features).
       The aggregation invariant is NOT enforced during optimization—gradient
       updates modify internal nodes independently. Dynamic updates should NOT
       be used in this mode as they would overwrite learned values.

    **Empty tree representation:**

    After all voxels are removed, the tree may be empty (``count == 0``). Invariants:
    - ``codes``, ``data``, ``children_mask``, ``weights`` are empty tensors (size 0)
    - ``structure`` has ``capacity >= 1`` with all entries = -1
    - ``maximum_depth`` remains defined (preserves resolution setting)
    - ``count == 0``

    Construction is NOT differentiable (discrete structure).
    Query operations (:func:`octree_sample`, :func:`octree_ray_marching`) support autograd.

    Examples
    --------
    >>> points = torch.rand(1000, 3) * 2 - 1  # [-1, 1]³
    >>> data = torch.rand(1000, 8)  # 8-dim features
    >>> tree = octree(points, data, maximum_depth=8)
    >>> tree.count
    tensor(...)  # Leaves + internal nodes
    >>> (tree.children_mask == 0).sum()  # Leaf count
    tensor(...)
    """

    codes: Tensor
    data: Tensor
    structure: Tensor
    children_mask: Tensor
    weights: Tensor
    maximum_depth: Tensor
    count: Tensor
    aggregation: Tensor


def octree(
    points: Tensor,
    data: Tensor,
    *,
    maximum_depth: int = 8,
    capacity_factor: float = 2.0,
    aggregation: str = "mean",
) -> Octree:
    """Build an octree from points in normalized [-1, 1]³ space.

    Parameters
    ----------
    points : Tensor, shape (n, 3)
        Point coordinates in [-1, 1]³.
    data : Tensor, shape (n, *value_shape)
        Data associated with each point.
    maximum_depth : int, default=8
        Tree depth. Resolution = 2^maximum_depth per axis.
        Depth 8 = 256³, depth 10 = 1024³, depth 12 = 4096³.
    capacity_factor : float, default=2.0
        Initial hash table size = estimated_total_nodes * capacity_factor.
        Total nodes includes both leaves and internal nodes.

        **Max displacement guarantee:** Construction tracks maximum probe
        displacement during insertion. If any entry exceeds max_probes (64),
        the table is rebuilt with 2x capacity and re-inserted. This repeats
        until all entries are within the probe bound, guaranteeing O(1)
        bounded lookups with no false negatives.
    aggregation : str, default="mean"
        How to compute internal node data from children: "mean", "sum", "max".
        Also used when multiple input points map to the same leaf voxel.

    Returns
    -------
    Octree
        Sparse voxel structure with full hierarchy.

    Notes
    -----
    **Full hierarchy construction:** The constructor builds the complete tree
    from root (depth 0) to leaves (maximum_depth). Internal nodes at each
    depth store aggregated data from their descendants.

    **Leaf data:** Points are quantized to voxel coordinates at maximum_depth.
    Multiple points in the same leaf voxel have their data aggregated using
    the specified aggregation function. The ``weights`` tensor tracks how many
    points contributed to each leaf.

    **Internal node data:** Computed via weighted aggregation up the tree.
    The ``weights`` tensor for internal nodes equals the sum of child weights,
    enabling proper weighted averaging.

    **Out-of-bounds handling:** Points outside [-1, 1]³ are silently clamped
    to the boundary.

    **Construction is NOT differentiable** (discrete structure).

    Examples
    --------
    >>> points = torch.rand(1000, 3) * 2 - 1  # [-1, 1]³
    >>> data = torch.rand(1000, 8)  # 8-dim features
    >>> tree = octree(points, data, maximum_depth=8)
    >>> tree.count
    tensor(1203)  # Leaves + internal nodes
    >>> (tree.children_mask == 0).sum()  # Leaf count
    tensor(847)
    """
    if points.dim() != 2 or points.size(-1) != 3:
        raise RuntimeError(
            f"points must be shape (n, 3), got {tuple(points.shape)}"
        )
    if data.dim() < 1:
        raise RuntimeError(f"data must be at least 1D, got {data.dim()}D")
    if points.size(0) != data.size(0):
        raise RuntimeError(
            f"points and data must have same number of elements, "
            f"got {points.size(0)} and {data.size(0)}"
        )
    if maximum_depth < 1 or maximum_depth > 15:
        raise RuntimeError(
            f"maximum_depth must be in [1, 15], got {maximum_depth}"
        )
    if capacity_factor < 1.0:
        raise RuntimeError(
            f"capacity_factor must be >= 1.0, got {capacity_factor}"
        )

    # Map aggregation string to integer enum
    aggregation_map = {
        "mean": AGGREGATION_MEAN,
        "sum": AGGREGATION_SUM,
        "max": AGGREGATION_MAX,
    }
    if aggregation not in aggregation_map:
        raise RuntimeError(
            f"aggregation must be one of {list(aggregation_map.keys())}, "
            f"got '{aggregation}'"
        )
    aggregation_int = aggregation_map[aggregation]

    # Call C++ kernel
    # Returns: (codes, data, structure, children_mask, weights, maximum_depth, count)
    result = torch.ops.torchscience.octree_build(
        points,
        data,
        maximum_depth,
        capacity_factor,
        aggregation_int,
    )

    return Octree(
        codes=result[0],
        data=result[1],
        structure=result[2],
        children_mask=result[3],
        weights=result[4],
        maximum_depth=result[5],
        count=result[6],
        aggregation=torch.tensor(aggregation_int, dtype=torch.int64),
        batch_size=[],
    )
