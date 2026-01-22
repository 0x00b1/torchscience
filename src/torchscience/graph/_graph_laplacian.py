"""Graph Laplacian matrix computation."""

from typing import Literal

import torch
from torch import Tensor


def graph_laplacian(
    adjacency: Tensor,
    *,
    normalization: Literal[
        "combinatorial", "symmetric", "random_walk"
    ] = "combinatorial",
) -> Tensor:
    r"""
    Compute the graph Laplacian matrix.

    The graph Laplacian is a matrix representation that encodes the structure
    of a graph and is fundamental to spectral graph theory.

    **Combinatorial Laplacian** (default):

    .. math::
        L = D - A

    **Symmetric Normalized Laplacian**:

    .. math::
        L_{sym} = I - D^{-1/2} A D^{-1/2}

    **Random Walk Normalized Laplacian**:

    .. math::
        L_{rw} = I - D^{-1} A

    Where :math:`A` is the adjacency matrix, :math:`D` is the degree matrix
    (diagonal matrix of node degrees), and :math:`I` is the identity matrix.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(*, N, N)``. Can be weighted (non-negative).
        For undirected graphs, should be symmetric. Self-loops (diagonal entries)
        are included in the degree computation.
    normalization : {"combinatorial", "symmetric", "random_walk"}, default="combinatorial"
        Type of Laplacian normalization:

        - ``"combinatorial"``: Standard Laplacian :math:`L = D - A`
        - ``"symmetric"``: Normalized :math:`L_{sym} = I - D^{-1/2} A D^{-1/2}`
        - ``"random_walk"``: Normalized :math:`L_{rw} = I - D^{-1} A`

    Returns
    -------
    Tensor
        Laplacian matrix of shape ``(*, N, N)``.

    Examples
    --------
    Combinatorial Laplacian of a simple graph:

    >>> import torch
    >>> from torchscience.graph_theory import graph_laplacian
    >>> adj = torch.tensor([
    ...     [0., 1., 1.],
    ...     [1., 0., 1.],
    ...     [1., 1., 0.],
    ... ])
    >>> L = graph_laplacian(adj)
    >>> L
    tensor([[ 2., -1., -1.],
            [-1.,  2., -1.],
            [-1., -1.,  2.]])

    The eigenvalues of the combinatorial Laplacian are non-negative:

    >>> torch.linalg.eigvalsh(L)
    tensor([0.0000, 3.0000, 3.0000])

    Symmetric normalized Laplacian (eigenvalues in [0, 2]):

    >>> L_sym = graph_laplacian(adj, normalization="symmetric")
    >>> torch.linalg.eigvalsh(L_sym)
    tensor([-0.0000,  1.5000,  1.5000])

    Gradients flow through the Laplacian:

    >>> adj = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    >>> L = graph_laplacian(adj)
    >>> L.sum().backward()
    >>> adj.grad
    tensor([[-2.,  0.],
            [ 0., -2.]])

    Batched computation:

    >>> batch_adj = torch.stack([adj.detach(), adj.detach() * 2])
    >>> L_batch = graph_laplacian(batch_adj)
    >>> L_batch.shape
    torch.Size([2, 2, 2])

    Notes
    -----
    - The combinatorial Laplacian is positive semi-definite with smallest
      eigenvalue 0. The multiplicity of eigenvalue 0 equals the number of
      connected components.
    - The symmetric normalized Laplacian has eigenvalues in [0, 2].
    - For isolated nodes (degree 0), normalized Laplacians use a small
      epsilon to avoid division by zero.
    - This function supports autograd for all normalization types.

    References
    ----------
    .. [1] Chung, F. R. K. (1997). "Spectral Graph Theory".
           CBMS Regional Conference Series in Mathematics, 92.
    .. [2] von Luxburg, U. (2007). "A Tutorial on Spectral Clustering".
           Statistics and Computing, 17(4), 395-416.

    See Also
    --------
    scipy.sparse.csgraph.laplacian : SciPy implementation
    """
    if adjacency.dim() < 2:
        raise ValueError(
            f"graph_laplacian: adjacency must be at least 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"graph_laplacian: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )

    N = adjacency.size(-1)

    # Compute degree (sum of edge weights for each node)
    degree = adjacency.sum(dim=-1)  # (*, N)

    if normalization == "combinatorial":
        # L = D - A
        # Create diagonal degree matrix and subtract adjacency
        L = torch.diag_embed(degree) - adjacency

    elif normalization == "symmetric":
        # L_sym = I - D^{-1/2} A D^{-1/2}
        # Handle zero-degree nodes with small epsilon
        eps = torch.finfo(adjacency.dtype).eps
        degree_inv_sqrt = torch.where(
            degree > eps,
            degree.rsqrt(),
            torch.zeros_like(degree),
        )
        # D^{-1/2} A D^{-1/2} using broadcasting
        # (*, N, 1) * (*, N, N) * (*, 1, N)
        normalized_adj = (
            degree_inv_sqrt.unsqueeze(-1)
            * adjacency
            * degree_inv_sqrt.unsqueeze(-2)
        )
        # I - normalized_adj
        L = (
            torch.eye(N, dtype=adjacency.dtype, device=adjacency.device)
            - normalized_adj
        )

    elif normalization == "random_walk":
        # L_rw = I - D^{-1} A
        eps = torch.finfo(adjacency.dtype).eps
        degree_inv = torch.where(
            degree > eps,
            1.0 / degree,
            torch.zeros_like(degree),
        )
        # D^{-1} A
        normalized_adj = degree_inv.unsqueeze(-1) * adjacency
        # I - D^{-1} A
        L = (
            torch.eye(N, dtype=adjacency.dtype, device=adjacency.device)
            - normalized_adj
        )

    else:
        raise ValueError(
            f"graph_laplacian: normalization must be one of "
            f"'combinatorial', 'symmetric', 'random_walk', got '{normalization}'"
        )

    return L
