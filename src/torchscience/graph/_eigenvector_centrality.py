"""Eigenvector centrality algorithm implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _compute_eigenvector_differentiable(adjacency: Tensor) -> Tensor:
    """Compute eigenvector centrality with differentiable operations.

    Eigenvector centrality is the eigenvector corresponding to the largest
    eigenvalue of the adjacency matrix.
    """
    N = adjacency.size(-1)

    # Use eigendecomposition - torch.linalg.eigh is differentiable
    eigenvalues, eigenvectors = torch.linalg.eigh(adjacency)

    # Take eigenvector corresponding to largest eigenvalue (last column)
    max_eigenvec = eigenvectors[..., N - 1]

    # Choose sign so that eigenvector is mostly positive
    # For Perron-Frobenius theorem, the principal eigenvector of a
    # non-negative matrix has all non-negative components
    # Use sign of sum to determine if we need to flip
    sign = torch.sign(max_eigenvec.sum())
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    max_eigenvec = max_eigenvec * sign

    # Clamp to non-negative (small numerical errors can give tiny negatives)
    max_eigenvec = torch.clamp(max_eigenvec, min=0.0)

    # Normalize to unit norm
    norm = max_eigenvec.norm(dim=-1, keepdim=True)
    max_eigenvec = max_eigenvec / torch.clamp(norm, min=1e-12)

    return max_eigenvec


def eigenvector_centrality(adjacency: Tensor) -> Tensor:
    r"""
    Compute eigenvector centrality for all nodes in a graph.

    Eigenvector centrality measures node importance by considering the
    importance of neighbors. A node with high eigenvector centrality is
    connected to other nodes that also have high eigenvector centrality.

    .. math::
        \lambda c_i = \sum_j A_{ij} c_j

    where :math:`\lambda` is the largest eigenvalue of the adjacency matrix
    :math:`A`, and :math:`\mathbf{c}` is the corresponding eigenvector.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(*, N, N)``. The matrix should be
        symmetric for undirected graphs.

    Returns
    -------
    Tensor
        Eigenvector centrality scores of shape ``(*, N)``. Values are
        normalized to unit norm and non-negative.

    Examples
    --------
    >>> import torch
    >>> from torchscience.graph import eigenvector_centrality
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 1.0],
    ...     [1.0, 0.0, 1.0],
    ...     [1.0, 1.0, 0.0],
    ... ])
    >>> ec = eigenvector_centrality(adj)
    >>> ec.shape
    torch.Size([3])

    Complete graph - all nodes have equal centrality:

    >>> N = 5
    >>> adj = torch.ones(N, N) - torch.eye(N)
    >>> ec = eigenvector_centrality(adj)
    >>> torch.allclose(ec, ec[0].expand(N), rtol=1e-4)
    True

    Notes
    -----
    - **Complexity**: O(N^3) for eigendecomposition.
    - **Symmetric matrices**: Uses ``torch.linalg.eigh`` which requires
      symmetric input for correct results.
    - **Non-negative output**: Takes absolute value of eigenvector components
      since eigenvectors can be negated.

    References
    ----------
    .. [1] Bonacich, P. (1972). "Factoring and weighting approaches to status
           scores and clique identification". Journal of Mathematical
           Sociology, 2(1), 113-120.
    """
    if adjacency.dim() < 2:
        raise ValueError(
            f"eigenvector_centrality: adjacency must be at least 2D, "
            f"got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"eigenvector_centrality: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )

    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    # Handle batched input
    if adjacency.dim() > 2:
        batch_shape = adjacency.shape[:-2]
        N = adjacency.size(-1)
        flat_adj = adjacency.reshape(-1, N, N)
        results = []
        for i in range(flat_adj.size(0)):
            if flat_adj[i].requires_grad:
                results.append(
                    _compute_eigenvector_differentiable(flat_adj[i])
                )
            else:
                results.append(
                    torch.ops.torchscience.eigenvector_centrality(flat_adj[i])
                )
        result = torch.stack(results, dim=0)
        return result.reshape(*batch_shape, N)

    # Use differentiable Python implementation for gradient support
    if adjacency.requires_grad:
        return _compute_eigenvector_differentiable(adjacency)

    # Direct call to C++ for non-differentiable case (faster)
    return torch.ops.torchscience.eigenvector_centrality(adjacency)
