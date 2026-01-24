"""Kraft inequality for prefix-free codes."""

import torch
from torch import Tensor


def kraft_inequality(
    lengths: Tensor,
    *,
    alphabet_size: int = 2,
) -> Tensor:
    r"""Compute the Kraft sum for code lengths.

    The Kraft inequality states that for a uniquely decodable code over an
    alphabet of size D, the codeword lengths l₁, ..., lₙ must satisfy:

    .. math::

        \sum_{i=1}^{n} D^{-l_i} \leq 1

    Conversely, if lengths satisfy this inequality, a prefix-free code with
    those lengths exists.

    Parameters
    ----------
    lengths : Tensor
        Codeword lengths. Shape: ``(..., n_symbols)`` where ``...`` represents
        optional batch dimensions. Lengths should be non-negative integers.
    alphabet_size : int, default=2
        Size of the code alphabet (D). Default is 2 for binary codes.

    Returns
    -------
    Tensor
        Kraft sum :math:`\sum_i D^{-l_i}`. Shape is the input shape with the
        last dimension removed.

        - If sum ≤ 1: A prefix-free code with these lengths exists
        - If sum > 1: No uniquely decodable code with these lengths exists
        - If sum = 1: The code is complete (optimal)

    Examples
    --------
    >>> import torch
    >>> # Optimal binary code for 3 symbols: lengths [1, 2, 2]
    >>> lengths = torch.tensor([1, 2, 2])
    >>> kraft_inequality(lengths)
    tensor(1.)  # 0.5 + 0.25 + 0.25 = 1.0 (complete)

    >>> # Suboptimal code: lengths [2, 2, 2]
    >>> lengths = torch.tensor([2, 2, 2])
    >>> kraft_inequality(lengths)
    tensor(0.75)  # Room for more codewords

    >>> # Invalid code: lengths [1, 1, 1]
    >>> lengths = torch.tensor([1, 1, 1])
    >>> kraft_inequality(lengths)
    tensor(1.5)  # > 1, not uniquely decodable

    Notes
    -----
    - For optimal (minimum expected length) codes, the Kraft sum equals 1.
    - Huffman codes always achieve Kraft sum = 1.
    - The McMillan inequality shows that the same bound applies to all
      uniquely decodable codes, not just prefix-free codes.
    - Supports gradients (though lengths are typically integers).

    See Also
    --------
    huffman_lengths : Compute optimal code lengths.
    source_coding_bound : Shannon's source coding theorem.

    References
    ----------
    .. [1] Kraft, L. G. (1949). A device for quantizing, grouping, and coding
           amplitude-modulated pulses. M.S. thesis, MIT.
    .. [2] McMillan, B. (1956). Two inequalities implied by unique
           decipherability. IEEE Trans. Inform. Theory, 2(4), 115-116.
    """
    if not isinstance(lengths, Tensor):
        raise TypeError(
            f"lengths must be a Tensor, got {type(lengths).__name__}"
        )

    if lengths.dim() == 0:
        raise ValueError("lengths must have at least 1 dimension")

    if alphabet_size < 2:
        raise ValueError(
            f"alphabet_size must be at least 2, got {alphabet_size}"
        )

    # Compute D^{-l_i} for each length
    # Use float for computation
    lengths_float = lengths.float()
    d = float(alphabet_size)

    # D^{-l} = exp(-l * log(D))
    contributions = torch.pow(d, -lengths_float)

    # Sum over the last dimension (symbols)
    kraft_sum = contributions.sum(dim=-1)

    return kraft_sum
