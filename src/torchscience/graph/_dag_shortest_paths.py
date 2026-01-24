"""DAG shortest paths using topological sort."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.graph._topological_sort import CycleError


class _DAGShortestPathsFunction(torch.autograd.Function):
    """Autograd function for DAG shortest paths with implicit differentiation."""

    @staticmethod
    def forward(
        ctx,
        adjacency: Tensor,
        source: int,
    ) -> tuple[Tensor, Tensor]:
        # Call C++ operator
        distances, predecessors = torch.ops.torchscience.dag_shortest_paths(
            adjacency, source
        )

        # Save for backward
        ctx.save_for_backward(adjacency, distances, predecessors)
        ctx.source = source

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
            return grad_adj, None

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

        return grad_adj, None


def dag_shortest_paths(
    adjacency: Tensor,
    source: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute single-source shortest paths in a directed acyclic graph (DAG).

    This algorithm uses topological sorting to find shortest paths in O(V + E)
    time, which is faster than Dijkstra's algorithm for DAGs.

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
        graph contains a cycle (including self-loops).

    Examples
    --------
    Simple DAG:

    >>> import torch
    >>> from torchscience.graph import dag_shortest_paths
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 4.0],
    ...     [inf, 0.0, 2.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = dag_shortest_paths(adj, source=0)
    >>> dist
    tensor([0., 1., 3.])
    >>> pred
    tensor([-1,  0,  1])

    Gradient through shortest paths (implicit differentiation):

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 4.0],
    ...     [inf, 0.0, 2.0],
    ...     [inf, inf, 0.0],
    ... ], requires_grad=True)
    >>> dist, _ = dag_shortest_paths(adj, source=0)
    >>> dist.sum().backward()
    >>> adj.grad  # Gradient flows through shortest path edges
    tensor([[0., 2., 0.],
            [0., 0., 1.],
            [0., 0., 0.]])

    Notes
    -----
    - **Complexity**: O(V + E) time using topological sort, O(V) space.
      This is faster than Dijkstra's O((V + E) log V) for DAGs.
    - **DAG requirement**: The graph must be a directed acyclic graph.
      Use :func:`dijkstra` for general graphs with non-negative weights.
    - **Gradient computation**: Uses implicit differentiation through the
      shortest path tree. The gradient with respect to edge (u, v) is nonzero
      only if that edge is on the shortest path tree.

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009).
           "Introduction to Algorithms" (3rd ed.). MIT Press. Section 24.2.

    See Also
    --------
    dijkstra : Single-source shortest paths for general graphs
    bellman_ford : Single-source shortest paths with negative weights
    topological_sort : Compute topological ordering of a DAG
    """
    # Input validation
    if adjacency.dim() != 2:
        raise ValueError(
            f"dag_shortest_paths: adjacency must be 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(0) != adjacency.size(1):
        raise ValueError(
            f"dag_shortest_paths: adjacency must be square, "
            f"got {adjacency.size(0)} x {adjacency.size(1)}"
        )
    N = adjacency.size(0)
    if source < 0 or source >= N:
        raise ValueError(
            f"dag_shortest_paths: source must be in [0, {N - 1}], got {source}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Use autograd function for gradient support
    if adjacency.requires_grad:
        return _DAGShortestPathsFunction.apply(adjacency, source)

    # Direct call for non-differentiable case
    try:
        return torch.ops.torchscience.dag_shortest_paths(adjacency, source)
    except RuntimeError as e:
        # Convert C++ cycle error to Python CycleError
        msg = str(e)
        if "cycle" in msg.lower():
            raise CycleError(msg) from None
        raise
