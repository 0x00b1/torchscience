"""k-d tree implementation with L1 extent heuristic splitting."""

from __future__ import annotations

import math

import torch
from tensordict import tensorclass
from torch import Tensor


@tensorclass
class KdTree:
    """k-d tree spatial data structure.

    This class wraps the tree structure tensors returned by the C++ kernel.
    Use `kd_tree()` to construct instances.

    As a tensorclass, KdTree supports:
    - Automatic batching: indexing with `tree[0]` or `tree[:2]`
    - Device movement: `tree.to("cuda")` or `tree.cuda()`
    - Serialization: `torch.save(tree, path)` / `torch.load(path)`
    - Reshaping: `tree.reshape(2, 2)` for batch dimension manipulation

    Attributes
    ----------
    points : Tensor
        Original points, shape [..., n, d].
    split_dim : Tensor
        Split dimension per node (-1 for leaf), shape [..., n_nodes].
    split_val : Tensor
        Split value per node (matches input dtype), shape [..., n_nodes].
    left : Tensor
        Left child index (-1 for leaf), shape [..., n_nodes].
    right : Tensor
        Right child index (-1 for leaf), shape [..., n_nodes].
    indices : Tensor
        Point indices in leaf order, shape [..., n].
    leaf_starts : Tensor
        Start index in 'indices' per leaf, shape [..., n_leaves].
    leaf_counts : Tensor
        Number of points per leaf, shape [..., n_leaves].

    Notes
    -----
    Tree construction is NOT differentiable (discrete structure).
    Query operations (k_nearest_neighbors, range_search) support autograd.

    Examples
    --------
    >>> points = torch.randn(100, 3)
    >>> tree = kd_tree(points)
    >>> tree.to("cuda")  # Move to GPU
    KdTree(...)

    >>> # Batched trees support indexing
    >>> batched = kd_tree(torch.randn(4, 100, 3))
    >>> batched[0]  # First tree
    KdTree(...)
    """

    points: Tensor
    split_dim: Tensor
    split_val: Tensor
    left: Tensor
    right: Tensor
    indices: Tensor
    leaf_starts: Tensor
    leaf_counts: Tensor


def kd_tree(
    points: Tensor,
    *,
    leaf_size: int = 10,
) -> KdTree:
    """Build a k-d tree from points using L1 extent heuristic.

    Parameters
    ----------
    points : Tensor, shape [..., n, d]
        Points to index. Last two dimensions are (n_points, n_dims).
        Leading dimensions become batch dimensions of the returned KdTree.
    leaf_size : int, default=10
        Maximum points per leaf node.

    Returns
    -------
    KdTree
        Tree structure. Batch dimensions match input leading dimensions.

    Notes
    -----
    Tree construction uses L1 extent heuristic with O(n·d) sweep per node
    for split selection, producing good tree quality with O(d) cost per
    split evaluation (vs O(d²) for surface area heuristic).

    **Differentiability:** Tree construction is NOT differentiable (discrete
    structure). Query operations (k_nearest_neighbors, range_search) support
    autograd through the returned distances.

    Examples
    --------
    >>> points = torch.randn(1000, 3)
    >>> tree = kd_tree(points, leaf_size=10)
    >>> tree.points.shape
    torch.Size([1000, 3])

    >>> # Batched construction
    >>> batched_points = torch.randn(4, 1000, 3)
    >>> tree = kd_tree(batched_points)
    >>> tree[0]  # First tree
    KdTree(...)
    """
    if points.dim() < 2:
        raise RuntimeError(
            f"points must be at least 2D [..., n, d], got {points.dim()}D"
        )
    if leaf_size <= 0:
        raise RuntimeError(f"leaf_size must be > 0, got {leaf_size}")

    *batch_dims, n, d = points.shape
    flat_batch = math.prod(batch_dims) if batch_dims else 1

    # Always work with batch dimension internally
    points_flat = points.reshape(flat_batch, n, d)

    # Single C++ call builds all trees in parallel and returns pre-padded tensors
    # Returns: (points, split_dim, split_val, left, right, indices, leaf_starts, leaf_counts)
    result = torch.ops.torchscience.kd_tree_build_batched(
        points_flat, leaf_size
    )

    # Build tree with flat batch dimension - no conditionals needed
    tree = KdTree(
        points=points_flat,
        split_dim=result[1],
        split_val=result[2],
        left=result[3],
        right=result[4],
        indices=result[5],
        leaf_starts=result[6],
        leaf_counts=result[7],
        batch_size=[flat_batch],
    )

    # Reshape to original batch dims, squeeze if unbatched
    if batch_dims:
        tree = tree.reshape(*batch_dims)
    else:
        tree = tree.squeeze(0)

    return tree
