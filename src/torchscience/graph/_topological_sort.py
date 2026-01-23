"""Topological sort for directed acyclic graphs (DAGs)."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


class CycleError(ValueError):
    """Raised when topological sort is attempted on a graph containing a cycle."""

    pass


def topological_sort(adjacency: Tensor) -> Tensor:
    r"""
    Compute a topological ordering of vertices in a directed acyclic graph (DAG).

    A topological ordering is a linear ordering of vertices such that for every
    directed edge (u, v), vertex u comes before vertex v in the ordering.
    Topological ordering is only possible for DAGs (directed graphs without cycles).

    This implementation uses Kahn's algorithm with O(V + E) time complexity.

    Parameters
    ----------
    adjacency : Tensor
        Adjacency matrix of shape ``(..., N, N)`` where ``adjacency[..., i, j]``
        is the edge weight from node ``i`` to node ``j``. Use ``float('inf')``
        for missing edges. An edge exists if the weight is finite and positive.
        Diagonal elements should be 0 (no self-loops).

    Returns
    -------
    order : Tensor
        Tensor of shape ``(..., N)`` with dtype ``int64`` containing the
        topological ordering. ``order[..., k]`` is the k-th vertex in the
        topological order.

    Raises
    ------
    ValueError
        If input is not at least 2D, not square, contains a cycle, or
        is not floating-point.

    Examples
    --------
    Simple linear DAG (0 -> 1 -> 2):

    >>> import torch
    >>> from torchscience.graph import topological_sort
    >>> inf = float("inf")
    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, 1.0],
    ...     [inf, inf, 0.0],
    ... ])
    >>> topological_sort(adj)
    tensor([0, 1, 2])

    Diamond DAG (0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3):

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, 1.0, inf],
    ...     [inf, 0.0, inf, 1.0],
    ...     [inf, inf, 0.0, 1.0],
    ...     [inf, inf, inf, 0.0],
    ... ])
    >>> order = topological_sort(adj)
    >>> order[0]  # Node 0 comes first
    tensor(0)
    >>> order[3]  # Node 3 comes last
    tensor(3)

    Cycle detection:

    >>> adj = torch.tensor([
    ...     [0.0, 1.0, inf],
    ...     [inf, 0.0, 1.0],
    ...     [1.0, inf, 0.0],  # Edge back to 0 creates cycle
    ... ])
    >>> topological_sort(adj)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: ... cycle ...

    Notes
    -----
    - **Algorithm**: Uses Kahn's algorithm (BFS-based) which has O(V + E) time
      complexity and O(V) space complexity.
    - **Uniqueness**: The topological ordering is not necessarily unique.
      This function returns one valid ordering.
    - **Edge convention**: An edge from i to j exists if ``adjacency[i, j]``
      is finite and positive. Zero values on the diagonal indicate no self-loop.
    - **Batched operation**: For batched input, each graph in the batch is
      processed independently.

    References
    ----------
    .. [1] Kahn, A. B. (1962). "Topological sorting of large networks".
           Communications of the ACM. 5 (11): 558-562.

    See Also
    --------
    scipy.sparse.csgraph : SciPy sparse graph algorithms
    networkx.topological_sort : NetworkX implementation
    """
    # Input validation
    if adjacency.dim() < 2:
        raise ValueError(
            f"topological_sort: adjacency must be at least 2D, got {adjacency.dim()}D"
        )
    if adjacency.size(-2) != adjacency.size(-1):
        raise ValueError(
            f"topological_sort: adjacency must be square, "
            f"got {adjacency.size(-2)} x {adjacency.size(-1)}"
        )
    if not adjacency.is_floating_point():
        raise ValueError(
            f"topological_sort: adjacency must be floating-point, "
            f"got {adjacency.dtype}"
        )

    # Handle sparse input
    if adjacency.is_sparse:
        adjacency = adjacency.to_dense()

    try:
        return torch.ops.torchscience.topological_sort(adjacency)
    except RuntimeError as e:
        # Convert C++ cycle error to Python CycleError
        msg = str(e)
        if "cycle" in msg.lower():
            raise CycleError(msg) from None
        raise
