"""Connected components algorithm implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def connected_components(
    adjacency: Tensor,
    *,
    directed: bool = True,
    connection: Literal["weak", "strong"] = "weak",
) -> tuple[int, Tensor]:
    r"""
    Find connected components of a graph.

    For undirected graphs, finds the maximal subsets of nodes where every
    pair of nodes is connected by a path.

    For directed graphs, supports two types of connectivity:

    - **Weak connectivity**: Ignores edge direction (treats graph as undirected)
    - **Strong connectivity**: Respects edge direction (u and v are in the same
      SCC if there exist paths u→v and v→u)

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(N, N)``. Entry ``adjacency[i, j]`` is
        nonzero if there is an edge from node ``i`` to node ``j``.
        Can be dense or sparse COO tensor.
    directed : bool, default=True
        If True, treat the graph as directed. If False, treat as undirected
        (only weak connectivity is meaningful).
    connection : {"weak", "strong"}, default="weak"
        Type of connectivity to find:

        - ``"weak"``: Ignore edge direction (connected if path exists
          ignoring direction)
        - ``"strong"``: Respect edge direction (strongly connected components
          via Tarjan's algorithm). Only valid when ``directed=True``.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : Tensor
        Component label for each node, shape ``(N,)``. Labels are integers
        in ``[0, n_components)``. Nodes with the same label are in the same
        component.

    Examples
    --------
    Undirected graph with two components:

    >>> import torch
    >>> from torchscience.graph import connected_components
    >>> # Two disconnected edges: 0-1 and 2-3
    >>> adj = torch.tensor([
    ...     [0., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...     [0., 0., 0., 1.],
    ...     [0., 0., 1., 0.],
    ... ])
    >>> n, labels = connected_components(adj, directed=False)
    >>> n
    2
    >>> labels
    tensor([0, 0, 1, 1])

    Directed graph - weak vs strong connectivity:

    >>> # Cycle: 0 -> 1 -> 2 -> 0
    >>> adj = torch.tensor([
    ...     [0., 1., 0.],
    ...     [0., 0., 1.],
    ...     [1., 0., 0.],
    ... ])
    >>> n_weak, _ = connected_components(adj, connection="weak")
    >>> n_weak  # All nodes reachable ignoring direction
    1
    >>> n_strong, _ = connected_components(adj, connection="strong")
    >>> n_strong  # All nodes in same SCC (cycle)
    1

    >>> # Chain: 0 -> 1 -> 2 (no back edges)
    >>> adj = torch.tensor([
    ...     [0., 1., 0.],
    ...     [0., 0., 1.],
    ...     [0., 0., 0.],
    ... ])
    >>> n_weak, _ = connected_components(adj, connection="weak")
    >>> n_weak
    1
    >>> n_strong, labels = connected_components(adj, connection="strong")
    >>> n_strong  # Each node is its own SCC (no back paths)
    3

    Single node:

    >>> adj = torch.tensor([[0.]])
    >>> n, labels = connected_components(adj)
    >>> n, labels
    (1, tensor([0]))

    Notes
    -----
    - **Weak connectivity** uses Union-Find (disjoint set union) with path
      compression and union by rank. Time complexity: O(N² α(N)) where α
      is the inverse Ackermann function.
    - **Strong connectivity** uses Tarjan's algorithm. Time complexity:
      O(N + E) where E is the number of edges.
    - Self-loops do not affect connectivity.
    - Isolated nodes (no edges) each form their own component.
    - **No autograd support**: This function returns discrete labels and
      does not support gradient computation.

    References
    ----------
    .. [1] Tarjan, R. E. (1972). "Depth-first search and linear graph
           algorithms". SIAM Journal on Computing, 1(2), 146-160.
    .. [2] Hopcroft, J., & Ullman, J. (1973). "Set Merging Algorithms".
           SIAM Journal on Computing, 2(4), 294-303.

    See Also
    --------
    scipy.sparse.csgraph.connected_components : SciPy implementation
    networkx.connected_components : NetworkX implementation
    """
    if adjacency.dim() < 2:
        raise ValueError(
            f"connected_components: adjacency must be at least 2D, "
            f"got {adjacency.dim()}D"
        )
    if adjacency.size(-1) != adjacency.size(-2):
        raise ValueError(
            f"connected_components: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )
    if connection not in ("weak", "strong"):
        raise ValueError(
            f"connected_components: connection must be 'weak' or 'strong', "
            f"got '{connection}'"
        )
    if connection == "strong" and not directed:
        raise ValueError(
            "connected_components: strong connectivity requires directed=True"
        )

    # Convert sparse to dense (C++ kernel operates on dense tensors)
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    n_components, labels = torch.ops.torchscience.connected_components(
        adjacency, directed, connection
    )

    return n_components, labels
