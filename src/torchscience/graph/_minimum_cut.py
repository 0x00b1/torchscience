"""Minimum cut operator using max-flow min-cut theorem."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class _MinimumCutFunction(torch.autograd.Function):
    """Autograd function for minimum cut.

    The gradient of cut_value with respect to capacity is 1 for edges in the
    minimum cut and 0 otherwise.
    """

    @staticmethod
    def forward(
        ctx,
        capacity: Tensor,
        source: int,
        sink: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Call C++ operator
        cut_value, reachable, cut_edges = torch.ops.torchscience.minimum_cut(
            capacity, source, sink
        )

        # Save for backward
        ctx.save_for_backward(capacity, reachable, cut_edges)
        ctx.source = source
        ctx.sink = sink

        return cut_value, reachable, cut_edges

    @staticmethod
    def backward(
        ctx,
        grad_cut_value: Tensor,
        grad_reachable: Tensor,
        grad_cut_edges: Tensor,
    ):
        capacity, reachable, cut_edges = ctx.saved_tensors

        grad_capacity = torch.zeros_like(capacity)

        # Gradient of cut_value w.r.t. capacity is 1 for cut edges, 0 elsewhere
        for i in range(cut_edges.shape[0]):
            u, v = cut_edges[i].tolist()
            grad_capacity[u, v] = grad_cut_value

        return grad_capacity, None, None


def minimum_cut(
    capacity: Tensor,
    source: int,
    sink: int,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute the minimum cut of a flow network using the max-flow min-cut theorem.

    The minimum cut is the smallest total capacity of edges that, if removed,
    would disconnect the source from the sink. By the max-flow min-cut theorem,
    the value of the minimum cut equals the maximum flow.

    .. math::
        \text{min-cut}(s, t) = \min_{(S, T) : s \in S, t \in T}
        \sum_{u \in S, v \in T} c_{u,v}

    where :math:`(S, T)` is a partition of vertices with source in :math:`S`
    and sink in :math:`T`.

    Parameters
    ----------
    capacity : Tensor
        Capacity matrix of shape ``(N, N)`` where ``capacity[i, j]`` is the
        maximum flow that can pass through edge ``(i, j)``. Non-positive values
        indicate no edge exists.
    source : int
        Index of the source vertex (0 to N-1).
    sink : int
        Index of the sink vertex (0 to N-1).

    Returns
    -------
    cut_value : Tensor
        Scalar tensor containing the minimum cut value (equals maximum flow).
    reachable : Tensor
        Boolean tensor of shape ``(N,)`` where ``reachable[i]`` is True if
        node ``i`` is reachable from the source in the residual graph after
        computing maximum flow.
    cut_edges : Tensor
        Tensor of shape ``(M, 2)`` containing the edges in the minimum cut,
        where ``M`` is the number of cut edges. Each row ``[u, v]`` represents
        an edge from ``u`` to ``v`` that crosses the cut.

    Raises
    ------
    ValueError
        If input is not 2D, not square, or source/sink is out of range.

    Examples
    --------
    Simple cut:

    >>> import torch
    >>> from torchscience.graph import minimum_cut
    >>> capacity = torch.tensor([
    ...     [0.0, 5.0],
    ...     [0.0, 0.0],
    ... ])
    >>> cut_value, reachable, cut_edges = minimum_cut(capacity, source=0, sink=1)
    >>> cut_value
    tensor(5.)
    >>> reachable
    tensor([ True, False])
    >>> cut_edges
    tensor([[0, 1]])

    Bottleneck edge:

    >>> capacity = torch.tensor([
    ...     [0.0, 10.0, 0.0],
    ...     [0.0, 0.0, 3.0],  # Bottleneck
    ...     [0.0, 0.0, 0.0],
    ... ])
    >>> cut_value, reachable, cut_edges = minimum_cut(capacity, source=0, sink=2)
    >>> cut_value
    tensor(3.)
    >>> cut_edges  # Cut is at edge (1, 2)
    tensor([[1, 2]])

    Gradient through cut value:

    >>> capacity = torch.tensor([
    ...     [0.0, 5.0],
    ...     [0.0, 0.0],
    ... ], requires_grad=True)
    >>> cut_value, _, _ = minimum_cut(capacity, source=0, sink=1)
    >>> cut_value.backward()
    >>> capacity.grad[0, 1]  # Gradient is 1 for cut edges
    tensor(1.)

    Notes
    -----
    - **Algorithm**: Uses the Edmonds-Karp algorithm (BFS-based Ford-Fulkerson)
      to compute maximum flow, then identifies the minimum cut by finding nodes
      reachable from source in the residual graph.
    - **Max-Flow Min-Cut Theorem**: The value of the minimum cut equals the
      maximum flow from source to sink.
    - **Cut Identification**: After max flow is computed, nodes reachable from
      source in the residual graph form set S. Cut edges are edges (u, v)
      where u is in S, v is not in S, and capacity[u, v] > 0.
    - **Time Complexity**: O(VE^2) where V is the number of vertices and E is
      the number of edges.

    References
    ----------
    .. [1] Ford, L. R., & Fulkerson, D. R. (1956). "Maximal flow through a
           network". Canadian Journal of Mathematics, 8, 399-404.
    .. [2] Edmonds, J., & Karp, R. M. (1972). "Theoretical improvements in
           algorithmic efficiency for network flow problems". Journal of the
           ACM, 19(2), 248-264.

    See Also
    --------
    edmonds_karp : Compute maximum flow and flow matrix
    """
    # Input validation
    if capacity.dim() != 2:
        raise ValueError(
            f"minimum_cut: capacity must be 2D, got {capacity.dim()}D"
        )
    if capacity.size(0) != capacity.size(1):
        raise ValueError(
            f"minimum_cut: capacity must be square, "
            f"got {capacity.size(0)} x {capacity.size(1)}"
        )
    N = capacity.size(0)
    if source < 0 or source >= N:
        raise ValueError(
            f"minimum_cut: source must be in [0, {N - 1}], got {source}"
        )
    if sink < 0 or sink >= N:
        raise ValueError(
            f"minimum_cut: sink must be in [0, {N - 1}], got {sink}"
        )

    # Handle sparse input
    if capacity.is_sparse:
        capacity = capacity.to_dense()

    # Use autograd function for gradient support
    if capacity.requires_grad:
        return _MinimumCutFunction.apply(capacity, source, sink)

    # Direct call for non-differentiable case
    return torch.ops.torchscience.minimum_cut(capacity, source, sink)
