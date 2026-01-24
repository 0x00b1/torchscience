"""Maximum bipartite matching using augmenting paths."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def maximum_bipartite_matching(
    biadjacency: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Find a maximum cardinality matching in a bipartite graph.

    A matching in a bipartite graph is a set of edges with no shared vertices.
    A maximum matching is one with the largest possible number of edges.
    This implementation uses the Hopcroft-Karp algorithm.

    Parameters
    ----------
    biadjacency : Tensor
        Biadjacency matrix of shape ``(*, M, N)`` where ``*`` is any number
        of batch dimensions, ``M`` is the size of the left partition and
        ``N`` is the size of the right partition.
        ``biadjacency[..., i, j] > 0`` indicates an edge between left node ``i``
        and right node ``j``. The actual values are ignored (only nonzero
        matters).

    Returns
    -------
    matching_size : Tensor
        Integer tensor with the size of the maximum matching. Shape is ``(*)``
        matching the batch dimensions of input.
    left_match : Tensor
        Tensor of shape ``(*, M)`` with dtype ``int64``.
        ``left_match[..., i]`` is the index of the right node matched to left
        node ``i``, or ``-1`` if left node ``i`` is unmatched.
    right_match : Tensor
        Tensor of shape ``(*, N)`` with dtype ``int64``.
        ``right_match[..., j]`` is the index of the left node matched to right
        node ``j``, or ``-1`` if right node ``j`` is unmatched.

    Raises
    ------
    ValueError
        If input is not at least 2D.

    Examples
    --------
    Simple bipartite graph:

    >>> import torch
    >>> from torchscience.graph import maximum_bipartite_matching
    >>> # 3 left nodes, 3 right nodes
    >>> # Edges: (0,0), (0,1), (1,1), (1,2), (2,2)
    >>> biadj = torch.tensor([
    ...     [1, 1, 0],
    ...     [0, 1, 1],
    ...     [0, 0, 1],
    ... ], dtype=torch.float32)
    >>> size, left, right = maximum_bipartite_matching(biadj)
    >>> size
    tensor(3)
    >>> left  # One possible matching
    tensor([0, 1, 2])
    >>> right
    tensor([0, 1, 2])

    Perfect matching (all nodes matched):

    >>> biadj = torch.tensor([
    ...     [1, 1],
    ...     [1, 1],
    ... ], dtype=torch.float32)
    >>> size, _, _ = maximum_bipartite_matching(biadj)
    >>> size
    tensor(2)

    No matching possible:

    >>> biadj = torch.zeros(2, 2)
    >>> size, left, right = maximum_bipartite_matching(biadj)
    >>> size
    tensor(0)
    >>> left
    tensor([-1, -1])

    Job assignment problem:

    >>> # 4 workers, 4 jobs
    >>> # workers[i] can do jobs where capable[i,j] = 1
    >>> capable = torch.tensor([
    ...     [1, 1, 0, 0],  # Worker 0 can do jobs 0, 1
    ...     [0, 1, 1, 0],  # Worker 1 can do jobs 1, 2
    ...     [0, 0, 1, 1],  # Worker 2 can do jobs 2, 3
    ...     [0, 0, 0, 1],  # Worker 3 can do job 3 only
    ... ], dtype=torch.float32)
    >>> size, assignment, _ = maximum_bipartite_matching(capable)
    >>> size  # All 4 workers can be assigned
    tensor(4)

    Notes
    -----
    - **Complexity**: O(E * sqrt(V)) time using Hopcroft-Karp algorithm,
      where V = M + N is total vertices and E is number of edges.
    - **Bipartite graphs**: The input represents a bipartite graph with
      left partition of size M and right partition of size N.
    - **Multiple matchings**: If multiple maximum matchings exist, the
      algorithm returns one of them (implementation-dependent).
    - **Weighted matching**: For weighted bipartite matching (assignment
      problem), use :func:`scipy.optimize.linear_sum_assignment` or
      implement the Hungarian algorithm.
    - **Differentiability**: This function is not differentiable since the
      output is discrete. For differentiable assignment, consider using
      :func:`torchscience.optimization.combinatorial.sinkhorn`.

    References
    ----------
    .. [1] Hopcroft, J. E.; Karp, R. M. (1973). "An n^{5/2} Algorithm for
           Maximum Matchings in Bipartite Graphs". SIAM Journal on Computing.
           2 (4): 225-231.
    .. [2] Kuhn, H. W. (1955). "The Hungarian Method for the assignment
           problem". Naval Research Logistics Quarterly. 2 (1-2): 83-97.

    Batched computation:

    >>> batch_biadj = torch.stack([biadj, biadj])  # (2, 3, 3)
    >>> size, left, right = maximum_bipartite_matching(batch_biadj)
    >>> size.shape
    torch.Size([2])

    See Also
    --------
    scipy.sparse.csgraph.maximum_bipartite_matching : SciPy implementation
    scipy.optimize.linear_sum_assignment : Weighted bipartite matching
    """
    # Input validation
    if biadjacency.dim() < 2:
        raise ValueError(
            f"maximum_bipartite_matching: biadjacency must be at least 2D, "
            f"got {biadjacency.dim()}D"
        )

    # Handle sparse input
    if biadjacency.is_sparse:
        biadjacency = biadjacency.to_dense()

    # Call C++ operator
    return torch.ops.torchscience.maximum_bipartite_matching(biadjacency)
