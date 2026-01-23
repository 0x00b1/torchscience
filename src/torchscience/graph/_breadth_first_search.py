"""Breadth-first search for graph traversal."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def breadth_first_search(
    adjacency: Tensor,
    source: int,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Perform breadth-first search (BFS) from a source node in a graph.

    BFS explores the graph level by level, visiting all nodes at distance k
    before visiting nodes at distance k+1. This function returns both the
    hop distances from the source and the predecessor of each node in the
    BFS tree.

    This implementation uses a queue-based algorithm with O(V + E) time complexity.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(..., N, N)`` where ``adjacency[..., i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. An edge exists if the weight is finite and positive.
        Diagonal elements should be 0 (no self-loops).
    source : int
        The source node index from which to start the BFS traversal.
        Must be in range ``[0, N)``.
    directed : bool, optional
        If ``True`` (default), treat the graph as directed (only consider edges
        from i to j based on ``adjacency[i, j]``). If ``False``, treat the graph
        as undirected (consider edge between i and j if either ``adjacency[i, j]``
        or ``adjacency[j, i]`` is finite and positive).

    Returns
    -------
    distances : Tensor
        Tensor of shape ``(..., N)`` with dtype ``int64`` containing the hop
        distances from the source node. ``distances[..., i]`` is the number of
        edges in the shortest path from source to node i. Unreachable nodes
        have distance ``-1``.
    predecessors : Tensor
        Tensor of shape ``(..., N)`` with dtype ``int64`` containing the
        predecessor of each node in the BFS tree. ``predecessors[..., i]`` is
        the node that precedes i on the shortest path from source.
        The source node and unreachable nodes have predecessor ``-1``.

    Raises
    ------
    ValueError
        If input is not at least 2D, not square, or not floating-point.
        If source is out of range.

    Examples
    --------
    Simple chain graph (0 -> 1 -> 2):

    >>> import torch
    >>> from torchscience.graph import breadth_first_search
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = breadth_first_search(adj, source=0)
    >>> dist
    tensor([0, 1, 2])
    >>> pred
    tensor([-1,  0,  1])

    Star graph with unreachable node:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, inf],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = breadth_first_search(adj, source=0)
    >>> dist[2]  # Node 2 is unreachable
    tensor(-1)

    Undirected traversal:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = breadth_first_search(adj, source=2, directed=False)
    >>> dist  # Can reach all nodes via undirected edges
    tensor([2, 1, 0])

    Notes
    -----
    - **Algorithm**: Uses a standard BFS queue with O(V + E) time complexity
      and O(V) space complexity.
    - **Unweighted distances**: BFS computes hop counts (number of edges),
      not weighted shortest paths. For weighted shortest paths, use
      :func:`dijkstra` or :func:`bellman_ford`.
    - **Edge convention**: An edge from i to j exists if ``adjacency[i, j]``
      is finite and positive. Zero values on the diagonal indicate no self-loop.
    - **Batched operation**: For batched input, each graph in the batch is
      processed independently with the same source node.

    References
    ----------
    .. [1] Cormen, T. H., et al. (2009). "Introduction to Algorithms",
           3rd Edition. MIT Press. Chapter 22.2.

    See Also
    --------
    dijkstra : Single-source shortest paths with non-negative weights
    bellman_ford : Single-source shortest paths with negative weights
    depth_first_search : Depth-first search traversal
    """
    # Input validation
    if adjacency.dim() < 2:
        raise ValueError(
            f"breadth_first_search: adjacency must be at least 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(-2) != adjacency.size(-1):
        raise ValueError(
            f"breadth_first_search: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )
    if not adjacency.is_floating_point():
        raise ValueError(
            f"breadth_first_search: adjacency must be floating-point, "
            f"got {adjacency.dtype}"
        )

    N = adjacency.size(-1)
    if source < 0 or source >= N:
        raise ValueError(
            f"breadth_first_search: source must be in range [0, {N}), got {source}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    return torch.ops.torchscience.breadth_first_search(
        adjacency, source, directed
    )
