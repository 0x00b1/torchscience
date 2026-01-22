"""Betweenness centrality algorithm implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def betweenness_centrality(
    adjacency: Tensor,
    normalized: bool = True,
) -> Tensor:
    r"""
    Compute betweenness centrality for all nodes in a graph.

    Betweenness centrality measures the extent to which a vertex lies on
    paths between other vertices. A node with high betweenness centrality
    has a large influence on the transfer of information through the network.

    .. math::
        c_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}

    where :math:`\sigma_{st}` is the total number of shortest paths from node
    :math:`s` to node :math:`t`, and :math:`\sigma_{st}(v)` is the number of
    those paths that pass through :math:`v`.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(*, N, N)``. Can be weighted or unweighted.
        For weighted graphs, edge weights represent distances (not capacities).
    normalized : bool, optional
        If True (default), normalize by ``2 / ((N-1)(N-2))`` for undirected
        graphs, yielding values in [0, 1]. If False, return raw counts.

    Returns
    -------
    Tensor
        Betweenness centrality scores of shape ``(*, N)``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.graph_theory import betweenness_centrality
    >>> # Star graph - center has highest betweenness
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 1.0, 1.0],
    ...     [1.0, 0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0, 0.0],
    ... ])
    >>> bc = betweenness_centrality(adj)
    >>> bc[0] > bc[1]  # Center node has highest centrality
    tensor(True)

    Chain graph - middle node has highest betweenness:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 0.0],
    ...     [1.0, 0.0, 1.0],
    ...     [0.0, 1.0, 0.0],
    ... ])
    >>> bc = betweenness_centrality(adj, normalized=False)
    >>> bc[1]  # Middle node is on path from 0 to 2
    tensor(1.)

    Notes
    -----
    - **Algorithm**: Uses Brandes' algorithm with O(VE) complexity for
      unweighted graphs.
    - **Undirected graphs**: Assumes the adjacency matrix is symmetric.
      For asymmetric matrices, the algorithm treats edges as bidirectional
      by considering the minimum of A[i,j] and A[j,i].
    - **Normalization**: For undirected graphs, the maximum possible
      betweenness is ``(N-1)(N-2)/2``, so normalization divides by this.
    - **Gradients**: Not differentiable due to discrete shortest path
      selection.

    References
    ----------
    .. [1] Brandes, U. (2001). "A faster algorithm for betweenness centrality."
           Journal of Mathematical Sociology, 25(2), 163-177.
    .. [2] Freeman, L.C. (1977). "A set of measures of centrality based on
           betweenness." Sociometry, 40(1), 35-41.
    """
    if adjacency.dim() < 2:
        raise ValueError(
            f"betweenness_centrality: adjacency must be at least 2D, "
            f"got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"betweenness_centrality: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )

    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Betweenness centrality is not differentiable due to discrete path selection
    # Always use the C++ kernel
    return torch.ops.torchscience.betweenness_centrality(adjacency, normalized)
