"""Huffman code lengths computation."""

import heapq

import torch
from torch import Tensor


def huffman_lengths(
    probabilities: Tensor,
    *,
    alphabet_size: int = 2,
) -> Tensor:
    r"""Compute optimal prefix-free code lengths using Huffman coding.

    Given symbol probabilities, computes the codeword lengths that minimize
    expected code length while satisfying the Kraft inequality.

    Mathematical Definition
    -----------------------
    For symbols with probabilities :math:`p_1, \ldots, p_n`, Huffman coding
    produces optimal prefix-free code lengths :math:`l_1, \ldots, l_n` that:

    1. Satisfy the Kraft inequality: :math:`\sum_i D^{-l_i} \leq 1`
    2. Minimize expected length: :math:`L = \sum_i p_i l_i`

    The expected length satisfies:
    :math:`H_D(X) \leq L < H_D(X) + 1`

    where :math:`H_D(X)` is the entropy in base D.

    Parameters
    ----------
    probabilities : Tensor
        Symbol probabilities. Shape: ``(..., n_symbols)`` where ``...``
        represents optional batch dimensions. Must sum to 1 along last dim.
    alphabet_size : int, default=2
        Size of the code alphabet (D). Default is 2 for binary codes.

    Returns
    -------
    Tensor
        Optimal code lengths. Same shape as input. Lengths are integers
        stored as float type.

    Examples
    --------
    >>> import torch
    >>> # Dyadic distribution: optimal lengths are exactly log_2(1/p)
    >>> probs = torch.tensor([0.5, 0.25, 0.25])
    >>> huffman_lengths(probs)
    tensor([1., 2., 2.])

    >>> # Uniform distribution: all lengths equal
    >>> probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> huffman_lengths(probs)
    tensor([2., 2., 2., 2.])

    Notes
    -----
    - Uses the standard heap-based Huffman algorithm.
    - The algorithm runs in O(n log n) time.
    - Does not support gradients (lengths are discrete).

    See Also
    --------
    kraft_inequality : Verify code length validity.
    source_coding_bound : Shannon's source coding theorem.
    shannon_entropy : Compute entropy lower bound.

    References
    ----------
    .. [1] Huffman, D. A. (1952). A Method for the Construction of
           Minimum-Redundancy Codes. Proceedings of the IRE, 40(9), 1098-1101.
    """
    if not isinstance(probabilities, Tensor):
        raise TypeError(
            f"probabilities must be a Tensor, got {type(probabilities).__name__}"
        )

    if probabilities.dim() == 0:
        raise ValueError("probabilities must have at least 1 dimension")

    if alphabet_size < 2:
        raise ValueError(
            f"alphabet_size must be at least 2, got {alphabet_size}"
        )

    if alphabet_size != 2:
        raise NotImplementedError(
            "Only binary (alphabet_size=2) is currently supported"
        )

    # Handle batch dimensions
    batch_shape = probabilities.shape[:-1]
    n_symbols = probabilities.shape[-1]

    if n_symbols == 0:
        return torch.empty_like(probabilities)

    if n_symbols == 1:
        return torch.zeros_like(probabilities)

    # Flatten batch dimensions
    probs_flat = probabilities.reshape(-1, n_symbols)
    batch_size = probs_flat.shape[0]

    # Compute lengths for each batch element
    lengths_list = []
    for i in range(batch_size):
        probs = probs_flat[i].tolist()
        lengths = _huffman_lengths_binary(probs)
        lengths_list.append(lengths)

    # Stack and reshape
    lengths_tensor = torch.tensor(
        lengths_list,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )

    if batch_shape:
        lengths_tensor = lengths_tensor.reshape(*batch_shape, n_symbols)
    else:
        lengths_tensor = lengths_tensor.squeeze(0)

    return lengths_tensor


def _huffman_lengths_binary(probs: list) -> list:
    """Compute binary Huffman lengths for a single distribution.

    Parameters
    ----------
    probs : list
        List of probabilities (floats)

    Returns
    -------
    list
        List of code lengths (floats)
    """
    n = len(probs)

    if n == 1:
        return [0.0]

    if n == 2:
        return [1.0, 1.0]

    # Build Huffman tree using a min-heap
    # Each element: (probability, unique_id, is_leaf, data)
    # data is symbol index for leaves, or (left, right) for internal nodes
    heap = []
    for i, p in enumerate(probs):
        heapq.heappush(heap, (p, i, i))  # (prob, tie-breaker, symbol_idx)

    next_id = n
    # Track parent and which child (0=left, 1=right) for each node
    parent = {}  # node_id -> (parent_id, child_idx)

    # Build tree by merging
    while len(heap) > 1:
        # Pop two smallest
        p1, id1, data1 = heapq.heappop(heap)
        p2, id2, data2 = heapq.heappop(heap)

        # Create merged node
        merged_prob = p1 + p2
        merged_id = next_id
        next_id += 1

        # Record parent relationships
        parent[id1] = (merged_id, 0)
        parent[id2] = (merged_id, 1)

        heapq.heappush(heap, (merged_prob, merged_id, None))

    # Compute depths (lengths) for each original symbol
    lengths = []
    for i in range(n):
        depth = 0
        node = i
        while node in parent:
            node, _ = parent[node]
            depth += 1
        lengths.append(float(depth))

    return lengths
