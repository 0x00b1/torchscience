"""Floyd-Warshall all-pairs shortest paths implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class _FloydWarshallFunction(torch.autograd.Function):
    """Autograd function for Floyd-Warshall with implicit differentiation.

    For Floyd-Warshall all-pairs shortest paths, the gradient can be computed
    via implicit differentiation. For each pair (i, j), the gradient of
    dist[i, j] with respect to the input adjacency matrix A is:
    - 1 for edges on the shortest path from i to j
    - 0 for edges not on the shortest path

    The gradient flows through the predecessor tensor - for each (i, j),
    we trace back the path using pred[i, j] and set gradient = 1 for
    those edges.
    """

    @staticmethod
    def forward(
        ctx,
        adjacency: Tensor,
        directed: bool,
    ) -> tuple[Tensor, Tensor]:
        # Call C++ operator
        distances, predecessors, has_negative_cycle = (
            torch.ops.torchscience.floyd_warshall(adjacency, directed)
        )

        # Save for backward
        ctx.save_for_backward(adjacency, distances, predecessors)
        ctx.directed = directed
        ctx.has_negative_cycle = has_negative_cycle

        return distances, predecessors, has_negative_cycle

    @staticmethod
    def backward(
        ctx, grad_distances: Tensor, grad_predecessors: Tensor, grad_flag
    ):
        adjacency, distances, predecessors = ctx.saved_tensors

        # Handle batched input
        batch_shape = adjacency.shape[:-2]
        N = adjacency.size(-1)

        # Initialize gradient for adjacency matrix
        grad_adj = torch.zeros_like(adjacency)

        # Flatten batch dimensions for easier iteration
        if batch_shape:
            flat_distances = distances.view(-1, N, N)
            flat_predecessors = predecessors.view(-1, N, N)
            flat_grad_distances = grad_distances.view(-1, N, N)
            flat_grad_adj = grad_adj.view(-1, N, N)
            batch_size = flat_distances.size(0)
        else:
            flat_distances = distances.unsqueeze(0)
            flat_predecessors = predecessors.unsqueeze(0)
            flat_grad_distances = grad_distances.unsqueeze(0)
            flat_grad_adj = grad_adj.unsqueeze(0)
            batch_size = 1

        # Process each graph in the batch
        for b in range(batch_size):
            dist = flat_distances[b]
            pred = flat_predecessors[b]
            grad_d = flat_grad_distances[b]
            g_adj = flat_grad_adj[b]

            # For each pair (i, j), trace the path from i to j and accumulate
            # gradient for edges along the path
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue

                    # Skip if no path exists (inf distance or pred == -1)
                    if torch.isinf(dist[i, j]) or pred[i, j] < 0:
                        continue

                    # Get the gradient for this (i, j) distance
                    g = grad_d[i, j]

                    # Skip if gradient is zero
                    if g == 0:
                        continue

                    # Trace back the path from j to i using predecessors
                    # pred[i, j] gives the node before j on the path from i to j
                    current = j
                    while pred[i, current].item() >= 0:
                        prev = pred[i, current].item()
                        # Edge (prev, current) is on the shortest path from i to j
                        g_adj[prev, current] += g
                        current = prev

        # For undirected graphs, we need to handle symmetry
        # The gradient with respect to adj[u, v] should include contributions
        # from both directions since the undirected graph uses min(adj[u,v], adj[v,u])
        if not ctx.directed:
            # Symmetrize gradient: each undirected edge receives gradient from both
            grad_adj = grad_adj + grad_adj.transpose(-1, -2)

        return grad_adj, None


class NegativeCycleError(ValueError):
    """Raised when the graph contains a negative cycle.

    The Floyd-Warshall algorithm cannot compute shortest paths when the
    graph contains a cycle with negative total weight, as paths can be
    made arbitrarily short by traversing the cycle repeatedly.
    """

    pass


def floyd_warshall(
    input: Tensor,
    *,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute all-pairs shortest paths using the Floyd-Warshall algorithm.

    The Floyd-Warshall algorithm computes the shortest path between every
    pair of vertices in a weighted graph. It works with both positive and
    negative edge weights, but the graph must not contain negative cycles.

    .. math::
        d_{ij}^{(k)} = \min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)})

    Parameters
    ----------
    input : Tensor
        Adjacency matrix of shape ``(*, N, N)`` where ``input[..., i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. Can be dense or sparse COO tensor.
    directed : bool, default=True
        If True, treat graph as directed. If False, symmetrize the adjacency
        matrix by taking the element-wise minimum of ``A`` and ``A.T``.

    Returns
    -------
    distances : Tensor
        Tensor of shape ``(*, N, N)`` with shortest path distances.
        ``distances[..., i, j]`` is the length of the shortest path from
        node ``i`` to node ``j``, or ``inf`` if no path exists.
    predecessors : Tensor
        Tensor of shape ``(*, N, N)`` with dtype ``int64``.
        ``predecessors[..., i, j]`` is the node immediately before ``j``
        on the shortest path from ``i`` to ``j``, or ``-1`` if no path exists.

    Raises
    ------
    NegativeCycleError
        If the graph contains a negative cycle.
    ValueError
        If input is not at least 2D, or last two dimensions are not equal.

    Examples
    --------
    Simple directed graph:

    >>> import torch
    >>> from torchscience.graph import floyd_warshall
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 4.0],
    ...     [inf, 0.0, 2.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, pred = floyd_warshall(adj)
    >>> dist
    tensor([[0., 1., 3.],
            [inf, 0., 2.],
            [inf, inf, 0.]])

    Path reconstruction (0 -> 2):

    >>> def reconstruct_path(pred, i, j):
    ...     if pred[i, j] == -1:
    ...         return [] if i != j else [i]
    ...     path = [j]
    ...     while pred[i, path[0]] != -1:
    ...         path.insert(0, pred[i, path[0]].item())
    ...     path.insert(0, i)
    ...     return path
    >>> reconstruct_path(pred, 0, 2)
    [0, 1, 2]

    Undirected graph (symmetrizes automatically):

    >>> adj_asym = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [2.0, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> dist, _ = floyd_warshall(adj_asym, directed=False)
    >>> dist[0, 1] == dist[1, 0]  # Symmetrized
    tensor(True)

    Batched computation:

    >>> batch_adj = torch.stack([adj, adj * 2])  # (2, 3, 3)
    >>> dist, pred = floyd_warshall(batch_adj)
    >>> dist.shape
    torch.Size([2, 3, 3])

    Notes
    -----
    - **Complexity**: O(N³) time, O(N²) space per graph.
    - **Sparse input**: Converted to dense internally. For very sparse graphs
      with many nodes, consider Dijkstra's algorithm instead.
    - **Negative weights**: Supported, but negative cycles cause an error.
    - **Path reconstruction**: Use the predecessors tensor to reconstruct
      paths. ``pred[i, j]`` gives the node before ``j`` on the shortest
      path from ``i`` to ``j``.
    - **CUDA**: Blocked/tiled algorithm for GPU acceleration.

    References
    ----------
    .. [1] Floyd, R. W. (1962). "Algorithm 97: Shortest Path".
           Communications of the ACM. 5 (6): 345.
    .. [2] Warshall, S. (1962). "A theorem on Boolean matrices".
           Journal of the ACM. 9 (1): 11-12.

    See Also
    --------
    scipy.sparse.csgraph.floyd_warshall : SciPy implementation
    """
    # Input validation
    if input.dim() < 2:
        raise ValueError(
            f"floyd_warshall: input must be at least 2D, got {input.dim()}D"
        )
    if input.size(-1) != input.size(-2):
        raise ValueError(
            f"floyd_warshall: last two dimensions must be equal, "
            f"got {input.size(-2)} x {input.size(-1)}"
        )
    if not input.is_floating_point():
        raise ValueError(
            f"floyd_warshall: input must be floating-point, got {input.dtype}"
        )

    # Handle sparse input
    if input.is_sparse:
        input = input.to_dense()

    # Use autograd function for gradient support
    if input.requires_grad:
        distances, predecessors, has_negative_cycle = (
            _FloydWarshallFunction.apply(input, directed)
        )
    else:
        # Direct call for non-differentiable case
        distances, predecessors, has_negative_cycle = (
            torch.ops.torchscience.floyd_warshall(input, directed)
        )

    if has_negative_cycle:
        raise NegativeCycleError(
            "floyd_warshall: graph contains a negative cycle"
        )

    return distances, predecessors
