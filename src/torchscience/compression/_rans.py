"""rANS (range Asymmetric Numeral Systems) encoding and decoding."""

import torch
from torch import Tensor


def rans_encode(
    symbols: Tensor,
    cdf: Tensor,
    precision: int = 16,
) -> tuple[list[int], int]:
    """Encode symbols using rANS (range Asymmetric Numeral Systems).

    ANS is a modern entropy coder that achieves near-optimal compression
    with excellent speed. The encoding produces output in reverse order,
    allowing streaming decoding in forward order.

    Parameters
    ----------
    symbols : Tensor
        Input symbols to encode. Shape: ``(n,)``. Must be 1-dimensional
        with integer values in range [0, len(cdf)-1).
    cdf : Tensor
        Cumulative distribution function. Shape: ``(n_symbols + 1,)``.
        cdf[0] should be 0, cdf[-1] should be 1 (or close to 1).
        cdf[i+1] - cdf[i] is the probability of symbol i.
    precision : int, default=16
        Bit precision for frequency counts.

    Returns
    -------
    output : list[int]
        Encoded output as list of 16-bit integers.
    num_words : int
        Number of 16-bit words in output.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([0, 1, 0, 0])
    >>> cdf = torch.tensor([0.0, 0.7, 1.0])  # P(0)=0.7, P(1)=0.3
    >>> output, n = rans_encode(symbols, cdf)

    Notes
    -----
    - Achieves compression rate approaching entropy H(X).
    - Not differentiable (discrete operation).
    - Faster than arithmetic coding due to simpler operations.
    - Output includes final state for decoding.

    See Also
    --------
    rans_decode : Decode rANS-coded data.
    """
    if not isinstance(symbols, Tensor):
        raise TypeError(
            f"symbols must be a Tensor, got {type(symbols).__name__}"
        )
    if not isinstance(cdf, Tensor):
        raise TypeError(f"cdf must be a Tensor, got {type(cdf).__name__}")

    if symbols.dim() != 1:
        raise ValueError(
            f"symbols must be 1-dimensional, got {symbols.dim()}D"
        )

    if symbols.numel() == 0:
        return [], 0

    # Convert CDF to integer frequencies
    total = 1 << precision
    cdf_int = (cdf.float() * total).long().tolist()
    cdf_int[-1] = total

    symbols_list = symbols.tolist()
    n_symbols = len(cdf_int) - 1

    # rANS state bounds
    # State must be in range [L, 2L) where L = 2^16
    L = 1 << 16
    MASK16 = 0xFFFF

    # Initialize state
    state = L

    # Output buffer (built in reverse)
    output = []

    # Encode in reverse order (rANS property)
    for symbol in reversed(symbols_list):
        if symbol < 0 or symbol >= n_symbols:
            raise ValueError(f"Symbol {symbol} out of range [0, {n_symbols})")

        # Get frequency and cumulative frequency
        freq_start = cdf_int[symbol]
        freq = cdf_int[symbol + 1] - freq_start

        # Renormalize: output bits while state is too large
        # We need state < freq * L after encoding
        max_state = freq << 16
        while state >= max_state:
            output.append(state & MASK16)
            state = state >> 16

        # Encode symbol into state
        # new_state = (state // freq) * total + (state % freq) + freq_start
        state = ((state // freq) << precision) + (state % freq) + freq_start

    # Output final state as two 16-bit words (high word first for decode order)
    output.append((state >> 16) & MASK16)  # high word
    output.append(state & MASK16)  # low word

    # Reverse output so it's in decode order
    output.reverse()

    return output, len(output)


def rans_decode(
    data: list[int],
    cdf: Tensor,
    length: int,
    precision: int = 16,
) -> Tensor:
    """Decode rANS-coded data.

    Parameters
    ----------
    data : list[int]
        Encoded data from rans_encode.
    cdf : Tensor
        Cumulative distribution function (same as used for encoding).
    length : int
        Number of symbols to decode.
    precision : int, default=16
        Bit precision (must match encoding precision).

    Returns
    -------
    Tensor
        Decoded symbols. Shape: ``(length,)``.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([0, 1, 0, 0])
    >>> cdf = torch.tensor([0.0, 0.7, 1.0])
    >>> output, n = rans_encode(symbols, cdf)
    >>> decoded = rans_decode(output, cdf, length=4)
    >>> torch.equal(decoded, symbols)
    True

    Notes
    -----
    - Not differentiable (discrete operation).

    See Also
    --------
    rans_encode : Encode symbols with rANS.
    """
    if not isinstance(cdf, Tensor):
        raise TypeError(f"cdf must be a Tensor, got {type(cdf).__name__}")

    if length == 0:
        return torch.empty(0, dtype=torch.long)

    # Convert CDF to integer frequencies
    total = 1 << precision
    cdf_int = (cdf.float() * total).long().tolist()
    cdf_int[-1] = total
    n_symbols = len(cdf_int) - 1

    # rANS constants
    L = 1 << 16
    mask_precision = total - 1

    # Read initial state from first two words (low word first, then high)
    data_idx = 0
    if len(data) >= 2:
        state = data[data_idx] | (data[data_idx + 1] << 16)
        data_idx = 2
    else:
        return torch.empty(0, dtype=torch.long)

    # Build symbol lookup table for fast decoding
    # symbol_table[freq_value] = symbol
    symbol_table = []
    for s in range(n_symbols):
        freq_start = cdf_int[s]
        freq_end = cdf_int[s + 1]
        for _ in range(freq_end - freq_start):
            symbol_table.append(s)

    decoded = []

    for _ in range(length):
        # Extract frequency position from state
        freq_pos = state & mask_precision

        # Find symbol using lookup table
        symbol = symbol_table[freq_pos]
        decoded.append(symbol)

        # Get frequency parameters
        freq_start = cdf_int[symbol]
        freq = cdf_int[symbol + 1] - freq_start

        # Decode: update state
        state = freq * (state >> precision) + freq_pos - freq_start

        # Renormalize: read bits while state is too small
        while state < L and data_idx < len(data):
            state = (state << 16) | data[data_idx]
            data_idx += 1

    return torch.tensor(decoded, dtype=torch.long)
