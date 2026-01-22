"""Bellman-Ford single-source shortest path algorithm."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class NegativeCycleError(ValueError):
    """Raised when the graph contains a negative cycle reachable from source.

    The Bellman-Ford algorithm cannot compute shortest paths when there is
    a negative cycle reachable from the source, as paths can be made
    arbitrarily short by traversing the cycle repeatedly.
    """

    pass


class _BellmanFordFunction(torch.autograd.Function):
    """Autograd function for Bellman-Ford with implicit differentiation."""

    @staticmethod
    def forward(
        ctx,
        adjacency: Tensor,
        source: int,
        directed: bool,
    ) -> tuple[Tensor, Tensor]:
        # Call C++ operator
        distances, predecessors, has_negative_cycle = (
            torch.ops.torchscience.bellman_ford(adjacency, source, directed)
        )

        if has_negative_cycle:
            raise NegativeCycleError(
                "bellman_ford: graph contains a negative cycle reachable from source"
            )

        # Save for backward
        ctx.save_for_backward(adjacency, distances, predecessors)
        ctx.source = source
        ctx.directed = directed

        return distances, predecessors

    @staticmethod
    def backward(ctx, grad_distances: Tensor, grad_predecessors: Tensor):
        adjacency, distances, predecessors = ctx.saved_tensors
        source = ctx.source

        # Implicit differentiation through shortest paths
        # Same as Dijkstra - gradient flows through shortest path tree
        N = adjacency.size(-1)
        grad_adj = torch.zeros_like(adjacency)

        # Get nodes sorted by distance (excluding source and unreachable)
        reachable_mask = ~torch.isinf(distances) & (
            torch.arange(N, device=distances.device) != source
        )
        reachable_indices = torch.where(reachable_mask)[0]

        if reachable_indices.numel() == 0:
            return grad_adj, None, None

        # Sort by decreasing distance (process furthest first)
        sorted_idx = torch.argsort(
            distances[reachable_indices], descending=True
        )
        sorted_nodes = reachable_indices[sorted_idx]

        # Accumulate gradients
        grad_d = grad_distances.clone()

        for node in sorted_nodes:
            pred = predecessors[node].item()
            if pred >= 0:
                # Gradient flows through the edge (pred -> node)
                grad_adj[pred, node] += grad_d[node]
                # Gradient accumulates to predecessor's distance
                grad_d[pred] += grad_d[node]

        # For undirected graphs, symmetrize gradient
        if not ctx.directed:
            grad_adj = grad_adj + grad_adj.T

        return grad_adj, None, None


def bellman_ford(
    adjacency: Tensor,
    source: int,
    *,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute single-source shortest paths using the Bellman-Ford algorithm.

    The Bellman-Ford algorithm finds the shortest paths from a source vertex
    to all other vertices in a weighted graph. Unlike Dijkstra's algorithm,
    it handles negative edge weights but requires O(VE) time.

    .. math::
        d_v^{(k)} = \min(d_v^{(k-1)}, \min_{u: (u,v) \in E}(d_u^{(k-1)} + w_{uv}))

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(N, N)`` where ``adjacency[i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges.
    source : int
        Index of the source vertex (0 to N-1).
    directed : bool, default=True
        If True, treat graph as directed. If False, symmetrize the adjacency
        matrix by taking the element-wise minimum of ``A`` and ``A.T``.

    Returns
    -------
    distances : Tensor
        Tensor of shape ``(N,)`` with shortest path distances from source.
        ``distances[i]`` is the length of the shortest path from source to
        node ``i``, or ``inf`` if no path exists.
    predecessors : Tensor
        Tensor of shape ``(N,)`` with dtype ``int64``.
        ``predecessors[i]`` is the node immediately before ``i`` on the
        shortest path from source to ``i``, or ``-1`` if no path exists
        or if ``i`` is the source.

    Raises
    ------
    NegativeCycleError
        If the graph contains a negative cycle reachable from source.
    ValueError
        If input is not 2D, not square, or source is out of range.

    Examples
    --------
    Graph with negative edges:

    >>> import torch
    >>> from torchscience.graph_theory import bellman_ford
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 4.0, inf],
    ...     [inf, 0.0, -2.0],  # Negative edge 1->2
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = bellman_ford(adj, source=0)
    >>> dist
    tensor([0., 4., 2.])
    >>> pred
    tensor([-1,  0,  1])

    Detecting negative cycles:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, -3.0],
    ...     [-1.0, inf, 0.0],  # Creates cycle with total weight -3
    ... ])
    >>> try:
    ...     bellman_ford(adj, source=0)
    ... except NegativeCycleError:
    ...     print("Negative cycle detected!")
    Negative cycle detected!

    Gradient through shortest paths:

    >>> adj = torch.tensor([
    ...     [0.0, 4.0, inf],
    ...     [inf, 0.0, -2.0],
    ...     [inf, inf, 0.0],
    ... ], requires_grad=True)
    >>> dist, _ = bellman_ford(adj, source=0)
    >>> dist.sum().backward()
    >>> adj.grad
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]])

    Notes
    -----
    - **Complexity**: O(VE) time, O(V) space where V is nodes, E is edges.
    - **Negative weights**: Supported, but negative cycles cause an error.
    - **When to use**: Use Bellman-Ford when edges can be negative. For
      non-negative weights, :func:`dijkstra` is faster (O((V+E) log V)).
    - **Gradient computation**: Uses implicit differentiation through the
      shortest path tree, same as Dijkstra.

    References
    ----------
    .. [1] Bellman, R. (1958). "On a routing problem". Quarterly of Applied
           Mathematics. 16: 87-90.
    .. [2] Ford, L. R. (1956). Network Flow Theory. Paper P-923.
           RAND Corporation.

    See Also
    --------
    dijkstra : Faster for non-negative weights
    floyd_warshall : All-pairs shortest paths
    scipy.sparse.csgraph.bellman_ford : SciPy implementation
    """
    # Input validation
    if adjacency.dim() != 2:
        raise ValueError(
            f"bellman_ford: adjacency must be 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(0) != adjacency.size(1):
        raise ValueError(
            f"bellman_ford: adjacency must be square, "
            f"got {adjacency.size(0)} x {adjacency.size(1)}"
        )
    N = adjacency.size(0)
    if source < 0 or source >= N:
        raise ValueError(
            f"bellman_ford: source must be in [0, {N - 1}], got {source}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Use autograd function for gradient support
    if adjacency.requires_grad:
        return _BellmanFordFunction.apply(adjacency, source, directed)

    # Direct call for non-differentiable case
    distances, predecessors, has_negative_cycle = (
        torch.ops.torchscience.bellman_ford(adjacency, source, directed)
    )

    if has_negative_cycle:
        raise NegativeCycleError(
            "bellman_ford: graph contains a negative cycle reachable from source"
        )

    return distances, predecessors
