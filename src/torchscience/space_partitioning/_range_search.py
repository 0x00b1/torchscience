"""Range search query returning nested tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from ._kd_tree import KdTree


def range_search(
    tree: KdTree,
    queries: Tensor,
    radius: float,
    *,
    p: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Find all neighbors within radius for each query point.

    Parameters
    ----------
    tree : KdTree
        Spatial index built by kd_tree().
    queries : Tensor, shape (m, d)
        Query points.
    radius : float
        Search radius (inclusive).
    p : float, default=2.0
        Minkowski p-norm (2.0 = Euclidean, 1.0 = Manhattan).

    Returns
    -------
    indices : Tensor (nested)
        Indices of neighbors per query. Use `.unbind()` to get list of tensors.
    distances : Tensor (nested)
        Distances to neighbors per query. Supports gradients w.r.t. query points.

    Notes
    -----
    Returns PyTorch nested tensors for variable-length results. Each query
    may have a different number of neighbors within the radius.

    Uses actual tree traversal with branch pruning for O(log n + k) average
    complexity per query, where k is the number of results.

    Examples
    --------
    >>> points = torch.randn(1000, 3)
    >>> tree = kd_tree(points)
    >>> queries = torch.randn(10, 3)
    >>> indices, distances = range_search(tree, queries, radius=1.0)
    >>> indices.is_nested
    True
    >>> # Access individual query results
    >>> for idx, dist in zip(indices.unbind(), distances.unbind()):
    ...     print(f"Found {len(idx)} neighbors")
    """
    if queries.dim() != 2:
        raise RuntimeError(f"queries must be 2D (m, d), got {queries.dim()}D")

    if radius < 0:
        raise RuntimeError(f"radius must be non-negative, got {radius}")

    if not isinstance(tree, KdTree):
        # Check if it's a TensorDict with _type field for compatibility
        tree_type = getattr(tree, "get", lambda *a: None)("_type", None)
        if tree_type is None:
            raise RuntimeError(f"Unsupported tree type: {type(tree).__name__}")
        if tree_type != "kd_tree":
            raise RuntimeError(f"Unsupported tree type: {tree_type}")

    points = tree.points
    d = points.size(1)

    if queries.size(1) != d:
        raise RuntimeError(
            f"Query dimension ({queries.size(1)}) must match "
            f"tree dimension ({d})"
        )

    return torch.ops.torchscience.range_search(
        tree.points,
        tree.split_dim,
        tree.split_val,
        tree.left,
        tree.right,
        tree.indices,
        tree.leaf_starts,
        tree.leaf_counts,
        queries,
        radius,
        p,
    )
