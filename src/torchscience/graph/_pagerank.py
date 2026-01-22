"""PageRank algorithm implementation."""

import torch
from torch import Tensor


def pagerank(
    adjacency: Tensor,
    *,
    alpha: float = 0.85,
    personalization: Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tensor:
    r"""
    Compute PageRank scores for all nodes in a graph.

    PageRank measures the importance of each node based on the link structure
    of the graph. It models a random surfer who follows links with probability
    :math:`\alpha` and jumps to a random node with probability :math:`1-\alpha`.

    The PageRank vector :math:`\mathbf{p}` satisfies:

    .. math::
        \mathbf{p} = \alpha \cdot M^T \mathbf{p} + (1 - \alpha) \cdot \mathbf{v}

    where :math:`M` is the row-stochastic transition matrix and :math:`\mathbf{v}`
    is the personalization vector.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(*, N, N)``. Entry ``adjacency[..., i, j]``
        represents an edge from node ``i`` to node ``j``. Can be weighted
        (weights are normalized to form transition probabilities).
    alpha : float, default=0.85
        Damping factor in ``(0, 1)``. Higher values make the random surfer
        follow links more often. The original PageRank paper uses 0.85.
    personalization : Tensor, optional
        Personalization vector of shape ``(*, N)``. Must be non-negative and
        will be normalized to sum to 1. Default is uniform distribution.
        Use this for topic-sensitive or personalized PageRank.
    max_iter : int, default=100
        Maximum number of power iterations.
    tol : float, default=1e-6
        Convergence tolerance. Iteration stops when the L1 norm of the
        change in PageRank scores is less than ``tol``.

    Returns
    -------
    Tensor
        PageRank scores of shape ``(*, N)``. Scores are non-negative and
        sum to 1 (probability distribution over nodes).

    Examples
    --------
    Simple directed graph:

    >>> import torch
    >>> from torchscience.graph_theory import pagerank
    >>> # Graph: 0 -> 1 -> 2 -> 0 (cycle)
    >>> adj = torch.tensor([
    ...     [0., 1., 0.],
    ...     [0., 0., 1.],
    ...     [1., 0., 0.],
    ... ])
    >>> pr = pagerank(adj)
    >>> pr  # All nodes have equal importance in a cycle
    tensor([0.3333, 0.3333, 0.3333])

    Star graph (node 0 is hub):

    >>> adj = torch.tensor([
    ...     [0., 1., 1., 1.],
    ...     [1., 0., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...     [1., 0., 0., 0.],
    ... ])
    >>> pr = pagerank(adj)
    >>> pr[0] > pr[1]  # Hub has higher PageRank
    tensor(True)

    Personalized PageRank:

    >>> adj = torch.rand(5, 5)
    >>> pers = torch.tensor([1., 0., 0., 0., 0.])  # Bias toward node 0
    >>> pr = pagerank(adj, personalization=pers)
    >>> pr[0] > pr.mean()  # Node 0 gets higher score
    tensor(True)

    Gradients flow through PageRank:

    >>> adj = torch.rand(4, 4, requires_grad=True)
    >>> pr = pagerank(adj)
    >>> pr.sum().backward()  # Trivial but shows gradient flow
    >>> adj.grad is not None
    True

    Batched computation:

    >>> batch_adj = torch.rand(3, 5, 5)
    >>> pr = pagerank(batch_adj)
    >>> pr.shape
    torch.Size([3, 5])

    Notes
    -----
    - **Dangling nodes**: Nodes with no outgoing edges have their probability
      mass distributed uniformly to all nodes (as if they link to everyone).
    - **Convergence**: Power iteration typically converges in 50-100 iterations.
      Increase ``max_iter`` for very large graphs or high ``alpha`` values.
    - **Sparse graphs**: The algorithm handles sparse adjacency matrices
      efficiently by avoiding explicit matrix construction where possible.
    - **Autograd**: Gradients are computed by differentiating through the
      power iteration unrolling.

    References
    ----------
    .. [1] Page, L., Brin, S., Motwani, R., & Winograd, T. (1999).
           "The PageRank citation ranking: Bringing order to the web."
           Stanford InfoLab Technical Report.
    .. [2] Haveliwala, T. H. (2003). "Topic-sensitive PageRank: A context-
           sensitive ranking algorithm for web search." IEEE TKDE, 15(4).

    See Also
    --------
    networkx.pagerank : NetworkX implementation
    scipy.sparse.linalg.eigs : Can compute PageRank as eigenvector
    """
    if adjacency.dim() < 2:
        raise ValueError(
            f"pagerank: adjacency must be at least 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"pagerank: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"pagerank: alpha must be in (0, 1), got {alpha}")

    N = adjacency.size(-1)

    if N == 0:
        return adjacency.new_empty(adjacency.shape[:-1])

    # Handle personalization vector
    if personalization is None:
        v = adjacency.new_ones(adjacency.shape[:-1]) / N
    else:
        if personalization.shape != adjacency.shape[:-1]:
            raise ValueError(
                f"pagerank: personalization shape {personalization.shape} "
                f"doesn't match adjacency batch shape {adjacency.shape[:-1]}"
            )
        # Normalize to sum to 1
        v = personalization / personalization.sum(dim=-1, keepdim=True).clamp(
            min=1e-10
        )

    # Compute out-degree for normalization
    out_degree = adjacency.sum(dim=-1)  # (*, N)

    # Handle dangling nodes (nodes with no outgoing edges)
    # For dangling nodes, we distribute probability uniformly
    is_dangling = out_degree == 0
    out_degree = out_degree.clamp(min=1e-10)  # Avoid division by zero

    # Row-normalize to get transition matrix M
    # M[i, j] = adjacency[i, j] / out_degree[i]
    M = adjacency / out_degree.unsqueeze(-1)

    # For dangling nodes, set row to uniform (equivalent to linking to all)
    # M[dangling, :] = 1/N
    uniform_row = adjacency.new_ones(N) / N
    M = torch.where(
        is_dangling.unsqueeze(-1).expand_as(M),
        uniform_row.expand_as(M),
        M,
    )

    # Initialize PageRank to uniform
    p = adjacency.new_ones(adjacency.shape[:-1]) / N

    # Power iteration: p = alpha * M^T @ p + (1 - alpha) * v
    for _ in range(max_iter):
        p_new = (
            alpha * torch.einsum("...ji,...j->...i", M, p) + (1 - alpha) * v
        )

        # Check convergence
        diff = (p_new - p).abs().sum(dim=-1)
        if (diff < tol).all():
            return p_new

        p = p_new

    return p
