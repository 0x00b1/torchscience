"""Minimum spanning tree algorithm using Prim's algorithm."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class _MinimumSpanningTreeFunction(torch.autograd.Function):
    """Autograd function for minimum spanning tree with implicit differentiation."""

    @staticmethod
    def forward(
        ctx,
        adjacency: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Call C++ operator
        total_weight, edges = torch.ops.torchscience.minimum_spanning_tree(
            adjacency
        )

        # Save for backward
        ctx.save_for_backward(adjacency, edges)

        return total_weight, edges

    @staticmethod
    def backward(ctx, grad_total_weight: Tensor, grad_edges: Tensor):
        adjacency, edges = ctx.saved_tensors

        # Implicit differentiation through MST
        # The edge weight is min(adj[u,v], adj[v,u]) for undirected treatment.
        # For gradient, we flow through the direction(s) that achieve the minimum.
        # If adj[u,v] < adj[v,u]: grad only to adj[u,v]
        # If adj[v,u] < adj[u,v]: grad only to adj[v,u]
        # If equal: split gradient 0.5 to each (subgradient)
        grad_adj = torch.zeros_like(adjacency)

        # edges has shape (N-1, 2) containing (u, v) pairs
        if edges.numel() > 0:
            for i in range(edges.size(0)):
                u, v = edges[i, 0].item(), edges[i, 1].item()
                if u < 0 or v < 0:  # Skip invalid edges (disconnected graph)
                    continue

                w_uv = adjacency[u, v]
                w_vu = adjacency[v, u]

                if w_uv < w_vu:
                    grad_adj[u, v] += grad_total_weight
                elif w_vu < w_uv:
                    grad_adj[v, u] += grad_total_weight
                else:
                    # Equal weights: split gradient
                    grad_adj[u, v] += 0.5 * grad_total_weight
                    grad_adj[v, u] += 0.5 * grad_total_weight

        return grad_adj


def minimum_spanning_tree(
    adjacency: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute the minimum spanning tree of a weighted undirected graph.

    The minimum spanning tree (MST) is a subset of edges that connects all
    vertices with the minimum total edge weight, without forming any cycles.
    This implementation uses Prim's algorithm.

    .. math::
        \text{MST} = \arg\min_{T \subseteq E, T \text{ spans } G} \sum_{(u,v) \in T} w_{uv}

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(N, N)`` where ``adjacency[i, j]``
        is the edge weight between node ``i`` and node ``j``. Use ``float('inf')``
        for missing edges. The graph is treated as undirected (uses minimum of
        ``adjacency[i, j]`` and ``adjacency[j, i]``).

    Returns
    -------
    total_weight : Tensor
        Scalar tensor with the total weight of the MST. Returns ``inf`` if the
        graph is not connected.
    edges : Tensor
        Tensor of shape ``(N-1, 2)`` with dtype ``int64`` containing the edges
        in the MST. Each row ``[u, v]`` represents an edge from node ``u`` to
        node ``v``. If graph is disconnected, returns edges of the spanning
        forest (fewer than N-1 edges, padded with -1).

    Raises
    ------
    ValueError
        If input is not 2D, not square, or has fewer than 1 node.

    Examples
    --------
    Simple triangle graph:

    >>> import torch
    >>> from torchscience.graph_theory import minimum_spanning_tree
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 3.0],
    ...     [1.0, 0.0, 2.0],
    ...     [3.0, 2.0, 0.0],
    ... ])
    >>> weight, edges = minimum_spanning_tree(adj)
    >>> weight
    tensor(3.)
    >>> edges  # MST uses edges (0,1) and (1,2) with total weight 1+2=3
    tensor([[0, 1],
            [1, 2]])

    Gradient through MST:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 3.0],
    ...     [1.0, 0.0, 2.0],
    ...     [3.0, 2.0, 0.0],
    ... ], requires_grad=True)
    >>> weight, edges = minimum_spanning_tree(adj)
    >>> weight.backward()
    >>> adj.grad  # Gradient is 1 for edges in MST, 0 otherwise
    tensor([[0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.]])

    Disconnected graph:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [1.0, 0.0, inf],
    ...     [inf, inf, 0.0],
    ... ])
    >>> weight, edges = minimum_spanning_tree(adj)
    >>> weight
    tensor(inf)

    Notes
    -----
    - **Complexity**: O((V + E) log V) time using Prim's algorithm with a
      priority queue, O(V) space.
    - **Undirected graphs**: The algorithm treats the input as undirected,
      using the minimum of ``adjacency[i, j]`` and ``adjacency[j, i]``.
    - **Disconnected graphs**: If the graph is not connected, returns ``inf``
      as total weight and the edges of the spanning forest.
    - **Gradient computation**: Uses implicit differentiation. The gradient
      with respect to edge (u, v) is equal to the gradient of the total weight
      if that edge is in the MST, zero otherwise.
    - **Ties**: When multiple edges have the same weight, the result depends
      on the order they are processed (implementation-dependent).

    References
    ----------
    .. [1] Prim, R. C. (1957). "Shortest connection networks and some
           generalizations". Bell System Technical Journal. 36 (6): 1389-1401.
    .. [2] Kruskal, J. B. (1956). "On the shortest spanning subtree of a graph
           and the traveling salesman problem". Proceedings of the American
           Mathematical Society. 7 (1): 48-50.

    See Also
    --------
    scipy.sparse.csgraph.minimum_spanning_tree : SciPy implementation
    """
    # Input validation
    if adjacency.dim() != 2:
        raise ValueError(
            f"minimum_spanning_tree: adjacency must be 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(0) != adjacency.size(1):
        raise ValueError(
            f"minimum_spanning_tree: adjacency must be square, "
            f"got {adjacency.size(0)} x {adjacency.size(1)}"
        )
    N = adjacency.size(0)
    if N < 1:
        raise ValueError(
            "minimum_spanning_tree: graph must have at least 1 node"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Use autograd function for gradient support
    if adjacency.requires_grad:
        return _MinimumSpanningTreeFunction.apply(adjacency)

    # Direct call for non-differentiable case
    return torch.ops.torchscience.minimum_spanning_tree(adjacency)
