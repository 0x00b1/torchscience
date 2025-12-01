"""Dynamic update operations for octree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._octree import Octree

# Aggregation mode constants (must match C++ enum)
AGGREGATION_MEAN = 0
AGGREGATION_SUM = 1
AGGREGATION_MAX = 2


def octree_insert(
    tree: "Octree",
    new_points: Tensor,
    new_data: Tensor,
    *,
    depth: int | None = None,
) -> "Octree":
    """Insert new voxels into the octree.

    Parameters
    ----------
    tree : Octree
        The tree to insert into.
    new_points : Tensor, shape (n, 3)
        Point coordinates in [-1, 1]^3 for new voxels.
    new_data : Tensor, shape (n, *value_shape)
        Data for new voxels. Must match tree.data shape except first dim.
    depth : int, optional
        Depth at which to insert new voxels.
        Defaults to tree.maximum_depth (leaves).

    Returns
    -------
    Octree
        New tree with inserted voxels.

    Notes
    -----
    **Graph break operation:** This rebuilds the tree structure and hash
    table. The returned Octree is a new object with new tensor buffers.

    **Ancestor creation:** If ancestors of new voxels don't exist, they
    are automatically created with aggregated data from descendants.

    **Conflict handling:** If inserting into a region that already has
    finer (deeper) voxels, the operation raises an error. Use
    :func:`octree_remove` or :func:`octree_merge` first to clear the region.

    **Duplicate handling:** If a point maps to an existing voxel at the
    target depth, data is aggregated using the tree's aggregation mode.

    Examples
    --------
    >>> tree = octree(points, data, maximum_depth=8)
    >>> new_pts = torch.tensor([[0.5, 0.5, 0.5]])
    >>> new_dat = torch.tensor([[1.0, 0.0, 0.0]])
    >>> tree = octree_insert(tree, new_pts, new_dat)
    """
    from ._octree import Octree

    if new_points.dim() != 2 or new_points.size(-1) != 3:
        raise RuntimeError(
            f"new_points must be shape (n, 3), got {tuple(new_points.shape)}"
        )
    if new_points.size(0) != new_data.size(0):
        raise RuntimeError(
            f"new_points and new_data must have same count, "
            f"got {new_points.size(0)} and {new_data.size(0)}"
        )

    maximum_depth = tree.maximum_depth.item()
    if depth is None:
        depth = maximum_depth
    if depth < 1 or depth > maximum_depth:
        raise RuntimeError(
            f"depth must be in [1, {maximum_depth}], got {depth}"
        )

    aggregation = tree.aggregation.item()

    result = torch.ops.torchscience.octree_insert(
        tree.codes,
        tree.data,
        tree.structure,
        tree.children_mask,
        tree.weights,
        new_points,
        new_data,
        depth,
        maximum_depth,
        aggregation,
    )

    return Octree(
        codes=result[0],
        data=result[1],
        structure=result[2],
        children_mask=result[3],
        weights=result[4],
        maximum_depth=result[5],
        count=result[6],
        aggregation=tree.aggregation.clone(),
        batch_size=[],
    )


def octree_remove(
    tree: "Octree",
    remove_codes: Tensor,
) -> "Octree":
    """Remove voxels from the octree by their Morton codes.

    Parameters
    ----------
    tree : Octree
        The tree to remove from.
    remove_codes : Tensor, shape (n,), dtype=int64
        Morton codes of voxels to remove.

    Returns
    -------
    Octree
        New tree with voxels removed.

    Notes
    -----
    **Graph break operation:** This rebuilds the tree structure and hash
    table. The returned Octree is a new object with new tensor buffers.

    **Ancestor pruning:** After removal, ancestors with no remaining
    children are automatically removed (pruned).

    **Non-existent codes:** Codes that don't exist in the tree are
    silently ignored.

    **Removing internal nodes:** If you remove an internal node, all its
    descendants are also removed.

    Examples
    --------
    >>> tree = octree(points, data, maximum_depth=8)
    >>> # Get codes of leaves to remove
    >>> leaf_mask = tree.children_mask == 0
    >>> codes_to_remove = tree.codes[leaf_mask][:10]
    >>> tree = octree_remove(tree, codes_to_remove)
    """
    from ._octree import Octree

    if remove_codes.dim() != 1:
        raise RuntimeError(
            f"remove_codes must be 1D, got {remove_codes.dim()}D"
        )
    if remove_codes.dtype != torch.int64:
        raise RuntimeError(
            f"remove_codes must be int64, got {remove_codes.dtype}"
        )

    maximum_depth = tree.maximum_depth.item()
    aggregation = tree.aggregation.item()

    result = torch.ops.torchscience.octree_remove(
        tree.codes,
        tree.data,
        tree.structure,
        tree.children_mask,
        tree.weights,
        remove_codes,
        maximum_depth,
        aggregation,
    )

    return Octree(
        codes=result[0],
        data=result[1],
        structure=result[2],
        children_mask=result[3],
        weights=result[4],
        maximum_depth=result[5],
        count=result[6],
        aggregation=tree.aggregation.clone(),
        batch_size=[],
    )


def octree_subdivide(
    tree: "Octree",
    subdivide_codes: Tensor,
) -> "Octree":
    """Subdivide leaf voxels into 8 children.

    Parameters
    ----------
    tree : Octree
        The tree containing voxels to subdivide.
    subdivide_codes : Tensor, shape (n,), dtype=int64
        Morton codes of leaf voxels to subdivide.

    Returns
    -------
    Octree
        New tree with subdivided voxels.

    Notes
    -----
    **Graph break operation:** This rebuilds the tree structure and hash
    table. The returned Octree is a new object with new tensor buffers.

    **Leaf requirement:** Only leaf voxels (children_mask == 0) can be
    subdivided. Attempting to subdivide internal nodes raises an error.

    **Depth limit:** Cannot subdivide voxels at maximum_depth since they
    have no room for children.

    **Data distribution:** Parent data is copied to all 8 children.
    Weight is distributed equally (parent_weight / 8).

    Examples
    --------
    >>> tree = octree(points, data, maximum_depth=8)
    >>> # Find coarse leaves (not at max depth) to subdivide
    >>> depths = (tree.codes >> 60) & 0xF
    >>> coarse_leaves = (tree.children_mask == 0) & (depths < 8)
    >>> codes_to_split = tree.codes[coarse_leaves][:5]
    >>> tree = octree_subdivide(tree, codes_to_split)
    """
    from ._octree import Octree

    if subdivide_codes.dim() != 1:
        raise RuntimeError(
            f"subdivide_codes must be 1D, got {subdivide_codes.dim()}D"
        )
    if subdivide_codes.dtype != torch.int64:
        raise RuntimeError(
            f"subdivide_codes must be int64, got {subdivide_codes.dtype}"
        )

    maximum_depth = tree.maximum_depth.item()
    aggregation = tree.aggregation.item()

    result = torch.ops.torchscience.octree_subdivide(
        tree.codes,
        tree.data,
        tree.structure,
        tree.children_mask,
        tree.weights,
        subdivide_codes,
        maximum_depth,
        aggregation,
    )

    return Octree(
        codes=result[0],
        data=result[1],
        structure=result[2],
        children_mask=result[3],
        weights=result[4],
        maximum_depth=result[5],
        count=result[6],
        aggregation=tree.aggregation.clone(),
        batch_size=[],
    )


def octree_merge(
    tree: "Octree",
    merge_codes: Tensor,
) -> "Octree":
    """Merge 8 sibling leaf voxels into their parent.

    Parameters
    ----------
    tree : Octree
        The tree containing voxels to merge.
    merge_codes : Tensor, shape (n,), dtype=int64
        Morton codes of parent voxels whose children should be merged.
        All 8 children must exist and be leaves.

    Returns
    -------
    Octree
        New tree with merged voxels.

    Notes
    -----
    **Graph break operation:** This rebuilds the tree structure and hash
    table. The returned Octree is a new object with new tensor buffers.

    **Complete sibling requirement:** All 8 children must exist and be
    leaves (children_mask == 0) for the merge to succeed. If any child
    is missing or is an internal node, the merge for that parent is skipped.

    **Data aggregation:** Child data is aggregated into the parent using
    the tree's aggregation mode (mean, sum, or max). For mean, weights
    are used for proper weighted averaging.

    **Parent becomes leaf:** After merge, the parent becomes a leaf
    (children_mask = 0) with aggregated data.

    Examples
    --------
    >>> tree = octree(points, data, maximum_depth=8)
    >>> # Find internal nodes at depth 7 (potential merge targets)
    >>> depths = (tree.codes >> 60) & 0xF
    >>> internal_d7 = (tree.children_mask != 0) & (depths == 7)
    >>> parent_codes = tree.codes[internal_d7][:3]
    >>> tree = octree_merge(tree, parent_codes)
    """
    from ._octree import Octree

    if merge_codes.dim() != 1:
        raise RuntimeError(f"merge_codes must be 1D, got {merge_codes.dim()}D")
    if merge_codes.dtype != torch.int64:
        raise RuntimeError(
            f"merge_codes must be int64, got {merge_codes.dtype}"
        )

    maximum_depth = tree.maximum_depth.item()
    aggregation = tree.aggregation.item()

    result = torch.ops.torchscience.octree_merge(
        tree.codes,
        tree.data,
        tree.structure,
        tree.children_mask,
        tree.weights,
        merge_codes,
        maximum_depth,
        aggregation,
    )

    return Octree(
        codes=result[0],
        data=result[1],
        structure=result[2],
        children_mask=result[3],
        weights=result[4],
        maximum_depth=result[5],
        count=result[6],
        aggregation=tree.aggregation.clone(),
        batch_size=[],
    )
