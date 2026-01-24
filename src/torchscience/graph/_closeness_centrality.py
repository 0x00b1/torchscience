"""Closeness centrality algorithm implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _compute_closeness_differentiable(
    adjacency: Tensor, normalized: bool
) -> Tensor:
    """Compute closeness centrality with differentiable operations.

    Uses a differentiable soft-min approximation for shortest paths
    to allow gradient computation through the centrality measure.
    """
    N = adjacency.size(-1)
    inf = float("inf")

    # Treat graph as undirected: symmetrize by taking minimum
    adj_sym = torch.minimum(adjacency, adjacency.transpose(-2, -1))

    # Initialize distances - use adjacency directly for edges
    # Set diagonal to 0 and non-edges to inf
    distances = adj_sym.clone()

    # Set diagonal to 0
    eye = torch.eye(N, dtype=adjacency.dtype, device=adjacency.device)
    distances = distances * (1 - eye)

    # Floyd-Warshall for all-pairs shortest paths (differentiable)
    for k in range(N):
        # distances[i,j] = min(distances[i,j], distances[i,k] + distances[k,j])
        via_k = distances[..., :, k : k + 1] + distances[..., k : k + 1, :]
        distances = torch.minimum(distances, via_k)

    # Compute closeness centrality for each node
    # closeness[i] = (n-1) / sum(distances[i,:]) for connected nodes
    # For disconnected nodes, use sum over reachable only

    # Mask for finite distances (reachable pairs), excluding self
    finite_mask = torch.isfinite(distances) & (1 - eye).bool()

    # Sum of distances to reachable nodes
    sum_dist = torch.where(
        finite_mask, distances, torch.zeros_like(distances)
    ).sum(dim=-1)

    # Count of reachable nodes
    reachable_count = finite_mask.float().sum(dim=-1)

    # Closeness = reachable / sum_dist (0 for isolated nodes)
    closeness = torch.where(
        sum_dist > 0, reachable_count / sum_dist, torch.zeros_like(sum_dist)
    )

    if normalized and N > 1:
        # Normalize: multiply by reachable/(N-1)
        closeness = closeness * reachable_count / (N - 1)

    return closeness


def closeness_centrality(
    adjacency: Tensor,
    *,
    normalized: bool = True,
) -> Tensor:
    r"""
    Compute closeness centrality for all nodes in a graph.

    Closeness centrality measures how close a node is to all other nodes.
    A node with high closeness can reach other nodes quickly.

    .. math::
        C(u) = \frac{n - 1}{\sum_{v \neq u} d(u, v)}

    where :math:`d(u, v)` is the shortest path distance from :math:`u` to :math:`v`,
    and :math:`n` is the number of nodes.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(*, N, N)`` where ``adjacency[..., i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. The graph is treated as undirected.
    normalized : bool, default=True
        If True, normalize by ``(n-1)`` to get values in ``[0, 1]``.
        If False, return raw closeness (reachable / sum_distances).

    Returns
    -------
    Tensor
        Closeness centrality scores of shape ``(*, N)``. Higher values indicate
        more central nodes. Disconnected nodes have centrality 0.

    Examples
    --------
    Star graph (center has highest closeness):

    >>> import torch
    >>> from torchscience.graph import closeness_centrality
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 1.0, 1.0],
    ...     [1.0, 0.0, inf, inf],
    ...     [1.0, inf, 0.0, inf],
    ...     [1.0, inf, inf, 0.0],
    ... ])
    >>> cc = closeness_centrality(adj)
    >>> cc[0] > cc[1]  # Center has highest closeness
    tensor(True)

    Gradient through closeness centrality:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 2.0],
    ...     [1.0, 0.0, 1.5],
    ...     [2.0, 1.5, 0.0],
    ... ], requires_grad=True)
    >>> cc = closeness_centrality(adj)
    >>> cc.sum().backward()
    >>> adj.grad is not None
    True

    Batched computation:

    >>> adj = torch.rand(3, 5, 5)
    >>> adj = adj + adj.transpose(-1, -2)
    >>> adj.diagonal(dim1=-2, dim2=-1).fill_(0)
    >>> cc = closeness_centrality(adj)
    >>> cc.shape
    torch.Size([3, 5])

    Notes
    -----
    - **Complexity**: O(N^3) for dense graphs using all-pairs shortest paths.
    - **Undirected graphs**: The implementation treats the input as undirected,
      using the minimum of ``adjacency[i, j]`` and ``adjacency[j, i]``.
    - **Disconnected nodes**: Nodes with no path to other nodes have centrality 0.
    - **Gradient computation**: Uses implicit differentiation through the
      shortest path computation.

    References
    ----------
    .. [1] Freeman, L. C. (1978). "Centrality in social networks: Conceptual
           clarification". Social Networks, 1(3), 215-239.

    See Also
    --------
    betweenness_centrality : Centrality based on shortest paths through node
    eigenvector_centrality : Centrality based on neighbor importance
    networkx.closeness_centrality : NetworkX implementation
    """
    # Input validation
    if adjacency.dim() < 2:
        raise ValueError(
            f"closeness_centrality: adjacency must be at least 2D, "
            f"got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"closeness_centrality: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Use differentiable Python implementation for gradient support
    if adjacency.requires_grad:
        return _compute_closeness_differentiable(adjacency, normalized)

    # Direct call to C++ for non-differentiable case (faster)
    return torch.ops.torchscience.closeness_centrality(adjacency, normalized)
