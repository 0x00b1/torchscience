import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def sinkhorn(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    *,
    epsilon: float = 0.1,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> Tensor:
    r"""
    Sinkhorn algorithm for entropy-regularized optimal transport.

    Computes the optimal transport plan between source distribution ``a``
    and target distribution ``b`` with cost matrix ``C``, regularized by
    entropy to encourage smooth solutions.

    Solves the optimization problem:

    .. math::

        \min_P \langle C, P \rangle + \epsilon H(P)
        \quad \text{s.t.} \quad P \mathbf{1} = a, \; P^T \mathbf{1} = b, \; P \geq 0

    where :math:`H(P) = -\sum_{ij} P_{ij} \log P_{ij}` is the entropy.

    Parameters
    ----------
    C : Tensor
        Cost matrix of shape ``(..., n, m)``. Entry ``C[i,j]`` is the cost
        of transporting mass from source ``i`` to target ``j``.
    a : Tensor
        Source marginal of shape ``(..., n)``. Must be non-negative and
        sum to 1 (probability distribution).
    b : Tensor
        Target marginal of shape ``(..., m)``. Must be non-negative and
        sum to 1 (probability distribution).
    epsilon : float, optional
        Entropy regularization strength. Smaller values give solutions
        closer to unregularized optimal transport but may converge slower.
        Default: 0.1.
    maxiter : int, optional
        Maximum number of Sinkhorn iterations. Default: 100.
    tol : float, optional
        Convergence tolerance on scaling vector change. Default: 1e-6.

    Returns
    -------
    Tensor
        Transport plan ``P`` of shape ``(..., n, m)``. Entry ``P[i,j]``
        is the amount of mass transported from source ``i`` to target ``j``.
        Satisfies ``P.sum(dim=-1) == a`` and ``P.sum(dim=-2) == b``.

    Examples
    --------
    Compute transport plan between uniform distributions:

    >>> C = torch.rand(3, 4)
    >>> a = torch.ones(3) / 3
    >>> b = torch.ones(4) / 4
    >>> P = sinkhorn(C, a, b)
    >>> P.sum(dim=-1)  # Should equal a
    tensor([0.3333, 0.3333, 0.3333])

    The transport cost (objective value) is:

    >>> (C * P).sum()
    tensor(...)

    Gradients flow through the transport plan:

    >>> C = torch.rand(3, 3, requires_grad=True)
    >>> a = torch.ones(3) / 3
    >>> b = torch.ones(3) / 3
    >>> P = sinkhorn(C, a, b)
    >>> (C * P).sum().backward()
    >>> C.grad.shape
    torch.Size([3, 3])

    Notes
    -----
    - The algorithm converges faster with larger ``epsilon`` but gives
      smoother (less sparse) transport plans.
    - For unregularized optimal transport (assignment problem), use
      ``epsilon < 0.01`` but increase ``maxiter``.
    - Gradients are only computed w.r.t. the cost matrix ``C``. The
      marginals ``a`` and ``b`` are treated as constants.

    References
    ----------
    - Cuturi, M. "Sinkhorn distances: Lightspeed computation of optimal
      transport." NeurIPS 2013.
    - Peyré, G. and Cuturi, M. "Computational optimal transport."
      Foundations and Trends in Machine Learning 11.5-6 (2019): 355-607.

    See Also
    --------
    https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem
    """
    return torch.ops.torchscience.sinkhorn(C, a, b, epsilon, maxiter, tol)
