"""Depth-first search for graph traversal."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def depth_first_search(
    adjacency: Tensor,
    source: int,
    directed: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform depth-first search (DFS) from a source node in a graph.

    DFS explores the graph by going as deep as possible along each branch
    before backtracking. This function returns the discovery time, finish time,
    and predecessor of each node in the DFS tree.

    This implementation uses an iterative stack-based algorithm with O(V + E)
    time complexity.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(..., N, N)`` where ``adjacency[..., i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. An edge exists if the weight is finite and positive.
        Diagonal elements should be 0 (no self-loops).
    source : int
        The source node index from which to start the DFS traversal.
        Must be in range ``[0, N)``.
    directed : bool, optional
        If ``True`` (default), treat the graph as directed (only consider edges
        from i to j based on ``adjacency[i, j]``). If ``False``, treat the graph
        as undirected (consider edge between i and j if either ``adjacency[i, j]``
        or ``adjacency[j, i]`` is finite and positive).

    Returns
    -------
    discovery_time : Tensor
        Tensor of shape ``(..., N)`` with dtype ``int64`` containing the
        discovery time of each node. ``discovery_time[..., i]`` is when node i
        was first visited during the DFS traversal. Unreachable nodes have
        discovery time ``-1``.
    finish_time : Tensor
        Tensor of shape ``(..., N)`` with dtype ``int64`` containing the
        finish time of each node. ``finish_time[..., i]`` is when the DFS
        finished exploring all of node i's descendants. Unreachable nodes
        have finish time ``-1``.
    predecessors : Tensor
        Tensor of shape ``(..., N)`` with dtype ``int64`` containing the
        predecessor of each node in the DFS tree. ``predecessors[..., i]`` is
        the node that discovered i during the DFS. The source node and
        unreachable nodes have predecessor ``-1``.

    Raises
    ------
    ValueError
        If input is not at least 2D, not square, or not floating-point.
        If source is out of range.

    Examples
    --------
    Simple chain graph (0 -> 1 -> 2):

    >>> import torch
    >>> from torchscience.graph import depth_first_search
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> disc, finish, pred = depth_first_search(adj, source=0)
    >>> disc
    tensor([0, 1, 2])
    >>> finish
    tensor([5, 4, 3])
    >>> pred
    tensor([-1,  0,  1])

    Unreachable node:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, inf],
    ...     [inf, inf, 0.0],
    ... ])
    >>> disc, finish, pred = depth_first_search(adj, source=0)
    >>> disc[2]  # Node 2 is unreachable
    tensor(-1)

    Undirected traversal:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> disc, finish, pred = depth_first_search(adj, source=2, directed=False)
    >>> disc  # All nodes reachable via undirected edges
    tensor([2, 1, 0])

    Notes
    -----
    - **Algorithm**: Uses an iterative DFS with explicit stack, achieving O(V + E)
      time complexity and O(V) space complexity.
    - **Discovery and Finish Times**: The discovery time is when a node is first
      visited, and the finish time is when all its descendants have been explored.
      These times satisfy the parenthesis theorem: for any two nodes u and v, either
      [disc[u], finish[u]] and [disc[v], finish[v]] are disjoint, or one contains
      the other.
    - **Edge convention**: An edge from i to j exists if ``adjacency[i, j]``
      is finite and positive. Zero values on the diagonal indicate no self-loop.
    - **Batched operation**: For batched input, each graph in the batch is
      processed independently with the same source node.

    References
    ----------
    .. [1] Cormen, T. H., et al. (2009). "Introduction to Algorithms",
           3rd Edition. MIT Press. Chapter 22.3.

    See Also
    --------
    breadth_first_search : Breadth-first search traversal
    topological_sort : Topological ordering of a DAG
    """
    # Input validation
    if adjacency.dim() < 2:
        raise ValueError(
            f"depth_first_search: adjacency must be at least 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(-2) != adjacency.size(-1):
        raise ValueError(
            f"depth_first_search: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )
    if not adjacency.is_floating_point():
        raise ValueError(
            f"depth_first_search: adjacency must be floating-point, "
            f"got {adjacency.dtype}"
        )

    N = adjacency.size(-1)
    if source < 0 or source >= N:
        raise ValueError(
            f"depth_first_search: source must be in range [0, {N}), got {source}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    return torch.ops.torchscience.depth_first_search(
        adjacency, source, directed
    )
