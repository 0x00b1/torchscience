"""Katz centrality algorithm implementation."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def _compute_katz_differentiable(
    adjacency: Tensor, alpha: float, beta: float, normalized: bool
) -> Tensor:
    """Compute Katz centrality with differentiable operations.

    Katz centrality: c = (I - alpha * A^T)^{-1} * beta * 1
    """
    N = adjacency.size(-1)

    # Direct solve: c = (I - alpha * A^T)^{-1} * beta * ones
    I = torch.eye(N, dtype=adjacency.dtype, device=adjacency.device)
    M = I - alpha * adjacency.t()
    b_vec = beta * torch.ones(
        N, 1, dtype=adjacency.dtype, device=adjacency.device
    )

    # Solve M * c = b_vec
    c = torch.linalg.solve(M, b_vec).squeeze(-1)

    if normalized:
        c = c / c.norm()

    return c


def katz_centrality(
    adjacency: Tensor,
    *,
    alpha: float = 0.1,
    beta: float = 1.0,
    normalized: bool = True,
) -> Tensor:
    r"""
    Compute Katz centrality for all nodes in a graph.

    Katz centrality measures node influence by counting walks of all lengths,
    with longer walks weighted less by factor :math:`\alpha`.

    .. math::
        c_i = \alpha \sum_j A_{ji} c_j + \beta

    Equivalently: :math:`\mathbf{c} = \beta (\mathbf{I} - \alpha \mathbf{A}^T)^{-1} \mathbf{1}`

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(*, N, N)``. Entry ``adjacency[..., i, j]``
        is nonzero if there is an edge from ``i`` to ``j``.
    alpha : float, default=0.1
        Attenuation factor in ``(0, 1/lambda_max)`` where ``lambda_max`` is the
        largest eigenvalue of the adjacency matrix. Smaller values weight
        local connections more heavily.
    beta : float, default=1.0
        Base centrality value for all nodes.
    normalized : bool, default=True
        If True, normalize result to unit norm.

    Returns
    -------
    Tensor
        Katz centrality scores of shape ``(*, N)``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.graph_theory import katz_centrality
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 1.0],
    ...     [1.0, 0.0, 1.0],
    ...     [1.0, 1.0, 0.0],
    ... ])
    >>> kc = katz_centrality(adj, alpha=0.2)
    >>> kc.shape
    torch.Size([3])

    Notes
    -----
    - **Convergence**: Requires ``alpha < 1/lambda_max`` for convergence.
    - **Complexity**: O(N^3) for matrix inversion.

    References
    ----------
    .. [1] Katz, L. (1953). "A new status index derived from sociometric
           analysis". Psychometrika, 18(1), 39-43.
    """
    if adjacency.dim() < 2:
        raise ValueError(
            f"katz_centrality: adjacency must be at least 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"katz_centrality: adjacency must be square, "
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
                    _compute_katz_differentiable(
                        flat_adj[i], alpha, beta, normalized
                    )
                )
            else:
                results.append(
                    torch.ops.torchscience.katz_centrality(
                        flat_adj[i], alpha, beta, normalized
                    )
                )
        result = torch.stack(results, dim=0)
        return result.reshape(*batch_shape, N)

    # Use differentiable Python implementation for gradient support
    if adjacency.requires_grad:
        return _compute_katz_differentiable(adjacency, alpha, beta, normalized)

    # Direct call to C++ for non-differentiable case (faster)
    return torch.ops.torchscience.katz_centrality(
        adjacency, alpha, beta, normalized
    )
