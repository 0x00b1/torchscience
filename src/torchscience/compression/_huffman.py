"""Huffman encoding and decoding."""

import heapq
from collections import Counter

import torch
from torch import Tensor


def huffman_encode(
    symbols: Tensor,
    probabilities: Tensor | None = None,
) -> tuple[list[int], dict[int, list[int]]]:
    """Encode symbols using Huffman coding.

    Huffman coding builds an optimal prefix-free code from symbol frequencies,
    assigning shorter codes to more frequent symbols.

    Parameters
    ----------
    symbols : Tensor
        Input symbols to encode. Shape: ``(n,)``. Must be 1-dimensional
        integer tensor.
    probabilities : Tensor or None, default=None
        Symbol probabilities. If None, computed from symbol frequencies
        in the input. Shape: ``(n_symbols,)`` where symbols are 0 to n_symbols-1.

    Returns
    -------
    bitstream : list[int]
        Encoded bits (0s and 1s).
    codebook : dict[int, list[int]]
        Mapping from symbol to its Huffman code (list of bits).

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([0, 0, 0, 1, 2])
    >>> bits, codebook = huffman_encode(symbols)
    >>> # Symbol 0 (most frequent) gets shortest code
    >>> len(codebook[0]) <= len(codebook[2])
    True

    Notes
    -----
    - Not differentiable (discrete operation).
    - The codebook is returned for use with huffman_decode.
    - For optimal compression, probabilities should match true frequencies.

    See Also
    --------
    huffman_decode : Decode Huffman-encoded bitstream.
    """
    if not isinstance(symbols, Tensor):
        raise TypeError(
            f"symbols must be a Tensor, got {type(symbols).__name__}"
        )

    if symbols.dim() != 1:
        raise ValueError(
            f"symbols must be 1-dimensional, got {symbols.dim()}D"
        )

    if symbols.numel() == 0:
        return [], {}

    symbols_list = symbols.tolist()

    # Compute probabilities from frequencies if not provided
    if probabilities is None:
        counts = Counter(symbols_list)
        total = len(symbols_list)
        unique_symbols = sorted(counts.keys())
        probs = {s: counts[s] / total for s in unique_symbols}
    else:
        if not isinstance(probabilities, Tensor):
            raise TypeError(
                f"probabilities must be a Tensor, got {type(probabilities).__name__}"
            )
        probs_list = probabilities.tolist()
        unique_symbols = list(range(len(probs_list)))
        probs = {i: p for i, p in enumerate(probs_list) if p > 0}
        unique_symbols = [s for s in unique_symbols if s in probs]

    if len(unique_symbols) == 0:
        return [], {}

    # Build Huffman tree
    codebook = _build_huffman_tree(probs)

    # Encode symbols
    bitstream = []
    for s in symbols_list:
        if s in codebook:
            bitstream.extend(codebook[s])
        else:
            # Symbol not in codebook (zero probability)
            raise ValueError(f"Symbol {s} not in codebook")

    return bitstream, codebook


def huffman_decode(
    bitstream: list[int],
    codebook: dict[int, list[int]],
    length: int | None = None,
) -> Tensor:
    """Decode a Huffman-encoded bitstream.

    Parameters
    ----------
    bitstream : list[int]
        Encoded bits (0s and 1s).
    codebook : dict[int, list[int]]
        Mapping from symbol to its Huffman code.
    length : int or None, default=None
        Number of symbols to decode. If None, decodes until bitstream is
        exhausted.

    Returns
    -------
    Tensor
        Decoded symbols. Shape: ``(n_decoded,)``.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([0, 0, 0, 1, 2])
    >>> bits, codebook = huffman_encode(symbols)
    >>> decoded = huffman_decode(bits, codebook, length=5)
    >>> torch.equal(decoded, symbols)
    True

    Notes
    -----
    - Not differentiable (discrete operation).
    - Decoding uses prefix-free property: no code is prefix of another.

    See Also
    --------
    huffman_encode : Encode symbols with Huffman coding.
    """
    if not codebook:
        return torch.empty(0, dtype=torch.long)

    # Build reverse lookup: code -> symbol
    # Since codes are prefix-free, we can use a trie-like approach
    reverse_lookup = {tuple(code): symbol for symbol, code in codebook.items()}

    decoded = []
    current_code = []
    bit_idx = 0

    while bit_idx < len(bitstream):
        if length is not None and len(decoded) >= length:
            break

        current_code.append(bitstream[bit_idx])
        bit_idx += 1

        code_tuple = tuple(current_code)
        if code_tuple in reverse_lookup:
            decoded.append(reverse_lookup[code_tuple])
            current_code = []

    return torch.tensor(decoded, dtype=torch.long)


def _build_huffman_tree(probs: dict[int, float]) -> dict[int, list[int]]:
    """Build Huffman tree and return codebook.

    Parameters
    ----------
    probs : dict[int, float]
        Symbol -> probability mapping.

    Returns
    -------
    dict[int, list[int]]
        Symbol -> code (list of bits) mapping.
    """
    if len(probs) == 1:
        # Single symbol case
        symbol = list(probs.keys())[0]
        return {symbol: [0]}

    # Build min-heap: (probability, unique_id, node)
    # node is either a symbol (int) or a tuple of children
    heap = []
    for i, (symbol, prob) in enumerate(probs.items()):
        heapq.heappush(heap, (prob, i, symbol))

    next_id = len(probs)

    # Build tree by merging
    while len(heap) > 1:
        prob1, _, node1 = heapq.heappop(heap)
        prob2, _, node2 = heapq.heappop(heap)

        merged_prob = prob1 + prob2
        merged_node = (node1, node2)  # (left, right)

        heapq.heappush(heap, (merged_prob, next_id, merged_node))
        next_id += 1

    # Build codebook from tree
    _, _, root = heap[0]
    codebook = {}
    _build_codes(root, [], codebook)

    return codebook


def _build_codes(
    node: int | tuple,
    prefix: list[int],
    codebook: dict[int, list[int]],
) -> None:
    """Recursively build codes from Huffman tree."""
    if isinstance(node, int):
        # Leaf node (symbol)
        codebook[node] = prefix if prefix else [0]
    else:
        # Internal node
        left, right = node
        _build_codes(left, prefix + [0], codebook)
        _build_codes(right, prefix + [1], codebook)
