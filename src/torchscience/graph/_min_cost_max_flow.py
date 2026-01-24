"""Minimum-cost maximum-flow algorithm."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class _MinCostMaxFlowFunction(torch.autograd.Function):
    """Autograd function for minimum-cost maximum-flow.

    Supports gradients with respect to both capacity and cost matrices.
    - Gradient w.r.t. capacity: Uses implicit differentiation via min-cut.
      For edges on the minimum cut, gradient is 1; for others, it's 0.
    - Gradient w.r.t. cost: The gradient is the flow on each edge.
    """

    @staticmethod
    def forward(
        ctx,
        capacity: Tensor,
        cost: Tensor,
        source: int,
        sink: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Call C++ operator
        max_flow, total_cost, flow = torch.ops.torchscience.min_cost_max_flow(
            capacity, cost, source, sink
        )

        # Save for backward
        ctx.save_for_backward(capacity, cost, flow)
        ctx.source = source
        ctx.sink = sink

        return max_flow, total_cost, flow

    @staticmethod
    def backward(
        ctx, grad_max_flow: Tensor, grad_total_cost: Tensor, grad_flow: Tensor
    ):
        capacity, cost, flow = ctx.saved_tensors
        source = ctx.source
        sink = ctx.sink

        N = capacity.size(0)
        grad_capacity = None
        grad_cost = None

        # Gradient w.r.t. cost: d(total_cost)/d(cost[i,j]) = flow[i,j]
        # since total_cost = sum_{i,j} flow[i,j] * cost[i,j]
        if ctx.needs_input_grad[1]:
            grad_cost = flow * grad_total_cost

        # Gradient w.r.t. capacity: same as max-flow (via min-cut)
        if ctx.needs_input_grad[0]:
            grad_capacity = torch.zeros_like(capacity)

            # Compute residual graph
            residual = capacity - flow
            # Add reverse flow capacities
            residual = residual + flow.T

            # Find minimum cut using BFS from source in residual graph
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

        return grad_capacity, grad_cost, None, None


def min_cost_max_flow(
    capacity: Tensor,
    cost: Tensor,
    source: int,
    sink: int,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute minimum-cost maximum-flow in a network.

    The minimum-cost maximum-flow algorithm finds the maximum flow from source
    to sink that has the minimum total cost among all maximum flows. It uses
    the successive shortest paths method (Ford-Fulkerson with Bellman-Ford
    for shortest path finding).

    .. math::
        \text{minimize} \sum_{(u,v) \in E} c_{u,v} \cdot f_{u,v}

    subject to:

    .. math::
        \begin{cases}
        0 \le f_{u,v} \le \text{capacity}_{u,v} & \text{(capacity constraint)} \\
        \sum_u f_{u,v} = \sum_w f_{v,w} & \text{(flow conservation)} \\
        \sum_v f_{s,v} = \text{max\_flow} & \text{(max flow constraint)}
        \end{cases}

    where :math:`c_{u,v}` is the cost per unit flow on edge :math:`(u,v)`,
    :math:`f_{u,v}` is the flow on edge :math:`(u,v)`, and :math:`s` is the source.

    Parameters
    ----------
    capacity : Tensor
        Capacity matrix of shape ``(N, N)`` where ``capacity[i, j]`` is the
        maximum flow that can pass through edge ``(i, j)``. Non-positive values
        indicate no edge exists.
    cost : Tensor
        Cost matrix of shape ``(N, N)`` where ``cost[i, j]`` is the cost per
        unit flow on edge ``(i, j)``. Must have the same shape as capacity.
    source : int
        Index of the source vertex (0 to N-1).
    sink : int
        Index of the sink vertex (0 to N-1).

    Returns
    -------
    max_flow : Tensor
        Scalar tensor containing the maximum flow value from source to sink.
    total_cost : Tensor
        Scalar tensor containing the minimum total cost to achieve max_flow.
        Computed as ``sum(flow[i,j] * cost[i,j])`` for all edges.
    flow : Tensor
        Flow matrix of shape ``(N, N)`` where ``flow[i, j]`` is the amount
        of flow on edge ``(i, j)``.

    Raises
    ------
    ValueError
        If inputs are not 2D, not square, have mismatched shapes, or
        source/sink is out of range.

    Examples
    --------
    Simple flow with costs:

    >>> import torch
    >>> from torchscience.graph import min_cost_max_flow
    >>> capacity = torch.tensor([
    ...     [0.0, 5.0],
    ...     [0.0, 0.0],
    ... ])
    >>> cost = torch.tensor([
    ...     [0.0, 2.0],
    ...     [0.0, 0.0],
    ... ])
    >>> max_flow, total_cost, flow = min_cost_max_flow(
    ...     capacity, cost, source=0, sink=1
    ... )
    >>> max_flow
    tensor(5.)
    >>> total_cost  # 5 units * cost 2 = 10
    tensor(10.)

    Cheaper path is preferred:

    >>> capacity = torch.tensor([
    ...     [0.0, 5.0, 5.0],
    ...     [0.0, 0.0, 5.0],
    ...     [0.0, 0.0, 0.0],
    ... ])
    >>> cost = torch.tensor([
    ...     [0.0, 1.0, 10.0],  # Path via node 1 is cheaper
    ...     [0.0, 0.0, 2.0],
    ...     [0.0, 0.0, 0.0],
    ... ])
    >>> max_flow, total_cost, flow = min_cost_max_flow(
    ...     capacity, cost, source=0, sink=2
    ... )
    >>> max_flow  # Max flow is 10 (5 via each path)
    tensor(10.)
    >>> total_cost  # 5*(1+2) + 5*10 = 15 + 50 = 65
    tensor(65.)

    Notes
    -----
    - **Algorithm**: Uses the successive shortest paths method, which
      repeatedly finds the shortest (minimum-cost) augmenting path using
      Bellman-Ford algorithm and augments flow along it. Time complexity
      is O(VE^2) in the worst case.
    - **Negative costs**: The algorithm handles reverse edges with negative
      costs in the residual graph, which is why Bellman-Ford is used
      instead of Dijkstra's algorithm.
    - **Relationship to max-flow**: The maximum flow value is the same as
      what would be computed by a pure max-flow algorithm (like Edmonds-Karp).
      The difference is that among all maximum flows, this algorithm finds
      the one with minimum cost.

    References
    ----------
    .. [1] Ford, L. R., & Fulkerson, D. R. (1956). "Maximal Flow Through a
           Network". Canadian Journal of Mathematics, 8, 399-404.
    .. [2] Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993).
           "Network Flows: Theory, Algorithms, and Applications". Prentice Hall.
    .. [3] Cormen, T. H., et al. (2009). "Introduction to Algorithms",
           3rd Edition. MIT Press. Chapter 26.

    See Also
    --------
    edmonds_karp : Maximum flow without cost consideration
    push_relabel : Alternative maximum flow algorithm
    bellman_ford : Shortest paths with negative edge weights
    """
    # Input validation
    if capacity.dim() != 2:
        raise ValueError(
            f"min_cost_max_flow: capacity must be 2D, got {capacity.dim()}D"
        )
    if capacity.size(0) != capacity.size(1):
        raise ValueError(
            f"min_cost_max_flow: capacity must be square, "
            f"got {capacity.size(0)} x {capacity.size(1)}"
        )
    if cost.dim() != 2:
        raise ValueError(
            f"min_cost_max_flow: cost must be 2D, got {cost.dim()}D"
        )
    if cost.size(0) != cost.size(1):
        raise ValueError(
            f"min_cost_max_flow: cost must be square, "
            f"got {cost.size(0)} x {cost.size(1)}"
        )
    if capacity.size(0) != cost.size(0) or capacity.size(1) != cost.size(1):
        raise ValueError(
            f"min_cost_max_flow: capacity and cost must have same shape, "
            f"got {capacity.size(0)} x {capacity.size(1)} vs "
            f"{cost.size(0)} x {cost.size(1)}"
        )
    N = capacity.size(0)
    if source < 0 or source >= N:
        raise ValueError(
            f"min_cost_max_flow: source must be in [0, {N - 1}], got {source}"
        )
    if sink < 0 or sink >= N:
        raise ValueError(
            f"min_cost_max_flow: sink must be in [0, {N - 1}], got {sink}"
        )

    # Handle sparse input
    if capacity.is_sparse:
        capacity = capacity.to_dense()
    if cost.is_sparse:
        cost = cost.to_dense()

    # Use autograd function for gradient support
    if capacity.requires_grad or cost.requires_grad:
        return _MinCostMaxFlowFunction.apply(capacity, cost, source, sink)

    # Direct call for non-differentiable case
    return torch.ops.torchscience.min_cost_max_flow(
        capacity, cost, source, sink
    )
