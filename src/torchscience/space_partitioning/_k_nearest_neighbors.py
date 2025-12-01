"""k-nearest neighbors query with tree traversal."""

from __future__ import annotations

import torch
from torch import Tensor

from ._kd_tree import KdTree


def k_nearest_neighbors(
    tree: KdTree,
    queries: Tensor,
    k: int,
    *,
    p: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Find k nearest neighbors for each query point using tree traversal.

    Parameters
    ----------
    tree : KdTree
        Spatial index built by kd_tree().
    queries : Tensor, shape (m, d)
        Query points.
    k : int
        Number of neighbors to find.
    p : float, default=2.0
        Minkowski p-norm (2.0 = Euclidean, 1.0 = Manhattan).

    Returns
    -------
    indices : Tensor, shape (m, k)
        Indices of k nearest neighbors per query.
    distances : Tensor, shape (m, k)
        Distances to k nearest neighbors per query.
        Supports first and second order gradients w.r.t. query points.

    Notes
    -----
    Uses actual tree traversal with branch pruning for O(log n) average
    complexity per query. Worst case is O(n) for pathological point
    distributions.

    Examples
    --------
    >>> points = torch.randn(1000, 3)
    >>> tree = kd_tree(points)
    >>> queries = torch.randn(10, 3)
    >>> indices, distances = k_nearest_neighbors(tree, queries, k=5)
    >>> indices.shape
    torch.Size([10, 5])
    """
    if queries.dim() != 2:
        raise RuntimeError(f"queries must be 2D (m, d), got {queries.dim()}D")

    if not isinstance(tree, KdTree):
        # Check if it's a TensorDict with _type field for compatibility
        tree_type = getattr(tree, "get", lambda *a: None)("_type", None)
        if tree_type is None:
            raise RuntimeError(f"Unsupported tree type: {type(tree).__name__}")
        if tree_type != "kd_tree":
            raise RuntimeError(f"Unsupported tree type: {tree_type}")

    points = tree.points
    n = points.size(0)
    d = points.size(1)

    if queries.size(1) != d:
        raise RuntimeError(
            f"Query dimension ({queries.size(1)}) must match "
            f"tree dimension ({d})"
        )

    if k <= 0 or k > n:
        raise RuntimeError(f"k ({k}) must be in [1, {n}]")

    return torch.ops.torchscience.k_nearest_neighbors(
        tree.points,
        tree.split_dim,
        tree.split_val,
        tree.left,
        tree.right,
        tree.indices,
        tree.leaf_starts,
        tree.leaf_counts,
        queries,
        k,
        p,
    )
