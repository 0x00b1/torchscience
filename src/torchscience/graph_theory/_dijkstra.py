"""Dijkstra's single-source shortest path algorithm."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class _DijkstraFunction(torch.autograd.Function):
    """Autograd function for Dijkstra with implicit differentiation."""

    @staticmethod
    def forward(
        ctx,
        adjacency: Tensor,
        source: int,
        directed: bool,
    ) -> tuple[Tensor, Tensor]:
        # Call C++ operator
        distances, predecessors = torch.ops.torchscience.dijkstra(
            adjacency, source, directed
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
        # For each node i, d[i] = d[pred[i]] + w[pred[i], i]
        # So grad_w[pred[i], i] = grad_d[i] and grad_d[pred[i]] += grad_d[i]
        #
        # We process nodes in reverse topological order (decreasing distance)
        # to accumulate gradients correctly

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


def dijkstra(
    adjacency: Tensor,
    source: int,
    *,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute single-source shortest paths using Dijkstra's algorithm.

    Dijkstra's algorithm finds the shortest paths from a source vertex to
    all other vertices in a weighted graph with non-negative edge weights.

    .. math::
        d_v = \min_{u: (u,v) \in E} (d_u + w_{uv})

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(N, N)`` where ``adjacency[i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. All weights must be non-negative.
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
    ValueError
        If input is not 2D, not square, source is out of range, or
        graph contains negative edge weights.

    Examples
    --------
    Simple directed graph:

    >>> import torch
    >>> from torchscience.graph_theory import dijkstra
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 4.0],
    ...     [inf, 0.0, 2.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = dijkstra(adj, source=0)
    >>> dist
    tensor([0., 1., 3.])
    >>> pred
    tensor([-1,  0,  1])

    Path reconstruction (0 -> 2):

    >>> def reconstruct_path(pred, source, target):
    ...     if pred[target] == -1 and target != source:
    ...         return []  # No path
    ...     path = [target]
    ...     while path[0] != source:
    ...         path.insert(0, pred[path[0]].item())
    ...     return path
    >>> reconstruct_path(pred, 0, 2)
    [0, 1, 2]

    Gradient through shortest paths (implicit differentiation):

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 4.0],
    ...     [inf, 0.0, 2.0],
    ...     [inf, inf, 0.0],
    ... ], requires_grad=True)
    >>> dist, _ = dijkstra(adj, source=0)
    >>> dist.sum().backward()
    >>> adj.grad  # Gradient flows through shortest path edges
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]])

    Notes
    -----
    - **Complexity**: O((N + E) log N) time using a priority queue, O(N) space.
    - **Non-negative weights**: Dijkstra requires all edge weights >= 0.
      For graphs with negative weights, use :func:`bellman_ford`.
    - **Gradient computation**: Uses implicit differentiation through the
      shortest path tree. The gradient with respect to edge (u, v) is nonzero
      only if that edge is on the shortest path tree.
    - **Sparse graphs**: For very sparse graphs, consider using sparse tensor
      input (converted to dense internally).

    References
    ----------
    .. [1] Dijkstra, E. W. (1959). "A note on two problems in connexion with
           graphs". Numerische Mathematik. 1: 269-271.

    See Also
    --------
    bellman_ford : Single-source shortest paths with negative weights
    floyd_warshall : All-pairs shortest paths
    scipy.sparse.csgraph.dijkstra : SciPy implementation
    """
    # Input validation
    if adjacency.dim() != 2:
        raise ValueError(
            f"dijkstra: adjacency must be 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(0) != adjacency.size(1):
        raise ValueError(
            f"dijkstra: adjacency must be square, "
            f"got {adjacency.size(0)} x {adjacency.size(1)}"
        )
    N = adjacency.size(0)
    if source < 0 or source >= N:
        raise ValueError(
            f"dijkstra: source must be in [0, {N - 1}], got {source}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Check for negative weights (excluding inf which is valid)
    # Skip for meta tensors since data-dependent operations don't work
    if adjacency.device.type != "meta":
        finite_mask = ~torch.isinf(adjacency)
        if (adjacency[finite_mask] < 0).any():
            raise ValueError(
                "dijkstra: graph contains negative edge weights. "
                "Use bellman_ford for graphs with negative weights."
            )

    # Use autograd function for gradient support
    if adjacency.requires_grad:
        return _DijkstraFunction.apply(adjacency, source, directed)

    # Direct call for non-differentiable case
    return torch.ops.torchscience.dijkstra(adjacency, source, directed)
