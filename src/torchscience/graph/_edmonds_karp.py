"""Edmonds-Karp maximum flow algorithm."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class _EdmondsKarpFunction(torch.autograd.Function):
    """Autograd function for Edmonds-Karp maximum flow with implicit differentiation.

    The gradient flows through the minimum cut edges. For edges on the minimum cut,
    the gradient of max_flow with respect to capacity is 1. For edges not on the
    minimum cut, the gradient is 0.
    """

    @staticmethod
    def forward(
        ctx,
        capacity: Tensor,
        source: int,
        sink: int,
    ) -> tuple[Tensor, Tensor]:
        # Call C++ operator
        max_flow, flow = torch.ops.torchscience.edmonds_karp(
            capacity, source, sink
        )

        # Save for backward
        ctx.save_for_backward(capacity, flow)
        ctx.source = source
        ctx.sink = sink

        return max_flow, flow

    @staticmethod
    def backward(ctx, grad_max_flow: Tensor, grad_flow: Tensor):
        capacity, flow = ctx.saved_tensors
        source = ctx.source
        sink = ctx.sink

        N = capacity.size(0)
        grad_capacity = torch.zeros_like(capacity)

        # Compute residual graph
        residual = capacity - flow
        # Add reverse flow capacities
        residual = residual + flow.T

        # Find minimum cut using BFS from source in residual graph
        # Edges crossing from source-side to sink-side are min-cut edges
        visited = torch.zeros(N, dtype=torch.bool, device=capacity.device)
        queue = [source]
        visited[source] = True

        while queue:
            u = queue.pop(0)
            for v in range(N):
                if not visited[v] and residual[u, v] > 1e-9:
                    visited[v] = True
                    queue.append(v)

        # Min-cut edges: from visited (source-side) to unvisited (sink-side)
        # with positive capacity
        for u in range(N):
            if visited[u]:
                for v in range(N):
                    if not visited[v] and capacity[u, v] > 0:
                        # This edge is on the min-cut
                        # Gradient of max_flow w.r.t. this capacity is 1
                        grad_capacity[u, v] = grad_max_flow

        return grad_capacity, None, None


def edmonds_karp(
    capacity: Tensor,
    source: int,
    sink: int,
) -> tuple[Tensor, Tensor]:
    r"""
    Compute maximum flow in a network using the Edmonds-Karp algorithm.

    The Edmonds-Karp algorithm is an implementation of the Ford-Fulkerson method
    that uses breadth-first search (BFS) to find augmenting paths. This guarantees
    O(VE^2) time complexity.

    .. math::
        \text{maximize} \sum_{v} f_{s,v} \quad \text{subject to} \quad
        \begin{cases}
        0 \le f_{u,v} \le c_{u,v} & \text{(capacity constraint)} \\
        \sum_u f_{u,v} = \sum_w f_{v,w} & \text{(flow conservation)}
        \end{cases}

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
    max_flow : Tensor
        Scalar tensor containing the maximum flow value from source to sink.
    flow : Tensor
        Flow matrix of shape ``(N, N)`` where ``flow[i, j]`` is the amount
        of flow on edge ``(i, j)``. The flow satisfies:
        - ``0 <= flow[i, j] <= capacity[i, j]`` (capacity constraint)
        - For all nodes except source and sink, inflow equals outflow
          (flow conservation)

    Raises
    ------
    ValueError
        If input is not 2D, not square, or source/sink is out of range.

    Examples
    --------
    Simple flow network:

    >>> import torch
    >>> from torchscience.graph import edmonds_karp
    >>> capacity = torch.tensor([
    ...     [0.0, 10.0, 10.0, 0.0],
    ...     [0.0, 0.0, 2.0, 4.0],
    ...     [0.0, 0.0, 0.0, 8.0],
    ...     [0.0, 0.0, 0.0, 0.0],
    ... ])
    >>> max_flow, flow = edmonds_karp(capacity, source=0, sink=3)
    >>> max_flow
    tensor(12.)

    Multiple paths:

    >>> capacity = torch.tensor([
    ...     [0.0, 5.0, 5.0, 0.0],
    ...     [0.0, 0.0, 0.0, 5.0],
    ...     [0.0, 0.0, 0.0, 5.0],
    ...     [0.0, 0.0, 0.0, 0.0],
    ... ])
    >>> max_flow, flow = edmonds_karp(capacity, source=0, sink=3)
    >>> max_flow  # Sum of two parallel paths
    tensor(10.)

    Gradient through max-flow (implicit differentiation via min-cut):

    >>> capacity = torch.tensor([
    ...     [0.0, 10.0, 0.0],
    ...     [0.0, 0.0, 3.0],  # Bottleneck edge
    ...     [0.0, 0.0, 0.0],
    ... ], requires_grad=True)
    >>> max_flow, _ = edmonds_karp(capacity, source=0, sink=2)
    >>> max_flow.backward()
    >>> capacity.grad[1, 2]  # Gradient is 1 for min-cut edges
    tensor(1.)

    Notes
    -----
    - **Algorithm**: Uses BFS to find augmenting paths, guaranteeing O(VE^2)
      time complexity where V is the number of vertices and E is the number
      of edges. This is an improvement over generic Ford-Fulkerson which may
      not terminate for irrational capacities.
    - **Max-Flow Min-Cut Theorem**: The maximum flow equals the minimum cut.
      The gradient computation uses this theorem: only edges on the minimum
      cut have non-zero gradients.
    - **Integer capacities**: For integer capacities, the algorithm runs in
      O(VE) augmenting path iterations, each taking O(E) time for BFS.
    - **Bidirectional edges**: The algorithm correctly handles networks with
      edges in both directions between the same pair of nodes.

    References
    ----------
    .. [1] Edmonds, J., & Karp, R. M. (1972). "Theoretical improvements in
           algorithmic efficiency for network flow problems". Journal of the
           ACM, 19(2), 248-264.
    .. [2] Cormen, T. H., et al. (2009). "Introduction to Algorithms",
           3rd Edition. MIT Press. Chapter 26.

    See Also
    --------
    push_relabel : Maximum flow using push-relabel algorithm (O(V^2 E))
    minimum_cut : Compute minimum cut directly
    """
    # Input validation
    if capacity.dim() != 2:
        raise ValueError(
            f"edmonds_karp: capacity must be 2D, got {capacity.dim()}D"
        )
    if capacity.size(0) != capacity.size(1):
        raise ValueError(
            f"edmonds_karp: capacity must be square, "
            f"got {capacity.size(0)} x {capacity.size(1)}"
        )
    N = capacity.size(0)
    if source < 0 or source >= N:
        raise ValueError(
            f"edmonds_karp: source must be in [0, {N - 1}], got {source}"
        )
    if sink < 0 or sink >= N:
        raise ValueError(
            f"edmonds_karp: sink must be in [0, {N - 1}], got {sink}"
        )

    # Handle sparse input
    if capacity.is_sparse:
        capacity = capacity.to_dense()

    # Use autograd function for gradient support
    if capacity.requires_grad:
        return _EdmondsKarpFunction.apply(capacity, source, sink)

    # Direct call for non-differentiable case
    return torch.ops.torchscience.edmonds_karp(capacity, source, sink)
