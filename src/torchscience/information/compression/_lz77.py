"""LZ77 dictionary-based compression."""

import torch
from torch import Tensor


def lz77_encode(
    symbols: Tensor,
    window_size: int = 4096,
    lookahead_size: int = 256,
) -> list[tuple[int, int, int]]:
    """Encode symbols using LZ77 dictionary compression.

    LZ77 encodes data by finding matches in a sliding window (dictionary)
    and outputting (offset, length, next_symbol) tuples.

    Parameters
    ----------
    symbols : Tensor
        Input symbols to encode. Shape: ``(n,)``. Must be 1-dimensional.
    window_size : int, default=4096
        Size of the sliding window (search buffer).
    lookahead_size : int, default=256
        Size of the lookahead buffer.

    Returns
    -------
    list[tuple[int, int, int]]
        List of (offset, length, next_symbol) tuples.
        - offset: Distance back in the window to the match start (0 if no match)
        - length: Length of the match (0 if no match)
        - next_symbol: The symbol following the match

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([1, 2, 3, 1, 2, 3, 4])
    >>> tokens = lz77_encode(symbols)
    >>> # First [1,2,3] has no history, then [1,2,3] matches offset 3
    >>> any(t[0] > 0 and t[1] > 0 for t in tokens)
    True

    Notes
    -----
    - Not differentiable (discrete operation).
    - Compression improves with repetitive data.
    - Larger window_size allows finding longer matches but uses more memory.

    See Also
    --------
    lz77_decode : Decode LZ77-compressed data.
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
        return []

    data = symbols.tolist()
    n = len(data)
    tokens = []
    pos = 0

    while pos < n:
        best_offset = 0
        best_length = 0

        # Search window: max(0, pos - window_size) to pos
        window_start = max(0, pos - window_size)

        # Lookahead buffer: pos to min(pos + lookahead_size, n)
        lookahead_end = min(pos + lookahead_size, n)

        # Find longest match in window
        for search_pos in range(window_start, pos):
            # Try to match starting at search_pos
            match_length = 0
            while (
                pos + match_length < lookahead_end
                and search_pos + match_length < pos
                and data[search_pos + match_length] == data[pos + match_length]
            ):
                match_length += 1

            # Allow copying beyond original match (run-length extension)
            while (
                pos + match_length < lookahead_end
                and data[search_pos + ((match_length) % (pos - search_pos))]
                == data[pos + match_length]
            ):
                match_length += 1

            if match_length > best_length:
                best_length = match_length
                best_offset = pos - search_pos

        # Output token
        if pos + best_length < n:
            next_symbol = data[pos + best_length]
        else:
            # End of data - use 0 as placeholder
            next_symbol = 0
            if best_length > 0:
                best_length = min(best_length, n - pos - 1)
                if best_length < 0:
                    best_length = 0
                if pos + best_length < n:
                    next_symbol = data[pos + best_length]

        # Handle case where we're at the very end
        if pos >= n:
            break

        if best_length == 0:
            # No match found - emit literal
            tokens.append((0, 0, data[pos]))
            pos += 1
        else:
            # Match found
            if pos + best_length >= n:
                # Adjust for end of data
                best_length = n - pos - 1
                if best_length < 0:
                    best_length = 0
                next_symbol = (
                    data[pos + best_length] if pos + best_length < n else 0
                )
            tokens.append((best_offset, best_length, next_symbol))
            pos += best_length + 1

    return tokens


def lz77_decode(
    tokens: list[tuple[int, int, int]],
) -> Tensor:
    """Decode LZ77-compressed data.

    Parameters
    ----------
    tokens : list[tuple[int, int, int]]
        List of (offset, length, next_symbol) tuples from lz77_encode.

    Returns
    -------
    Tensor
        Decoded symbols. Shape: ``(n,)``.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([1, 2, 3, 1, 2, 3, 4])
    >>> tokens = lz77_encode(symbols)
    >>> decoded = lz77_decode(tokens)
    >>> torch.equal(decoded, symbols)
    True

    Notes
    -----
    - Not differentiable (discrete operation).

    See Also
    --------
    lz77_encode : Encode data with LZ77 compression.
    """
    if not tokens:
        return torch.empty(0, dtype=torch.long)

    decoded = []

    for offset, length, next_symbol in tokens:
        if length > 0 and offset > 0:
            # Copy from history
            start_pos = len(decoded) - offset
            for i in range(length):
                # Handle run-length extension (copy index wraps)
                decoded.append(decoded[start_pos + (i % offset)])
        # Append the next symbol
        decoded.append(next_symbol)

    return torch.tensor(decoded, dtype=torch.long)
