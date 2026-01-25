"""Arithmetic encoding and decoding."""

import torch
from torch import Tensor


def arithmetic_encode(
    symbols: Tensor,
    cdf: Tensor,
    precision: int = 32,
) -> tuple[list[int], int]:
    """Encode symbols using arithmetic coding.

    Arithmetic coding achieves near-optimal compression by representing
    the entire message as a single number in [0, 1), subdividing the
    interval according to symbol probabilities.

    Parameters
    ----------
    symbols : Tensor
        Input symbols to encode. Shape: ``(n,)``. Must be 1-dimensional
        with integer values in range [0, len(cdf)-1).
    cdf : Tensor
        Cumulative distribution function. Shape: ``(n_symbols + 1,)``.
        cdf[0] should be 0, cdf[-1] should be 1 (or close to 1).
        cdf[i+1] - cdf[i] is the probability of symbol i.
    precision : int, default=32
        Bit precision for interval arithmetic. Higher precision allows
        longer messages but uses more memory.

    Returns
    -------
    bitstream : list[int]
        Encoded bits (0s and 1s).
    num_bits : int
        Number of significant bits in the output.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([0, 1, 0, 0])
    >>> cdf = torch.tensor([0.0, 0.7, 1.0])  # P(0)=0.7, P(1)=0.3
    >>> bits, n = arithmetic_encode(symbols, cdf)

    Notes
    -----
    - Achieves compression rate approaching entropy H(X).
    - Not differentiable (discrete operation).
    - Uses integer arithmetic internally for numerical stability.

    See Also
    --------
    arithmetic_decode : Decode arithmetic-coded bitstream.
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

    # Convert to integer arithmetic for precision
    # Scale CDF to integer range [0, 2^precision)
    max_range = 1 << precision
    half = 1 << (precision - 1)
    quarter = 1 << (precision - 2)

    # Convert CDF to integers
    cdf_int = (cdf.float() * max_range).long().tolist()

    symbols_list = symbols.tolist()
    n_symbols = len(cdf_int) - 1

    # Initialize interval [low, high)
    low = 0
    high = max_range

    # Pending bits for carry propagation
    pending_bits = 0
    bitstream = []

    def output_bit(bit: int) -> None:
        """Output a bit and any pending bits."""
        nonlocal pending_bits
        bitstream.append(bit)
        while pending_bits > 0:
            bitstream.append(1 - bit)
            pending_bits -= 1

    for symbol in symbols_list:
        if symbol < 0 or symbol >= n_symbols:
            raise ValueError(f"Symbol {symbol} out of range [0, {n_symbols})")

        # Narrow interval based on symbol
        range_size = high - low
        high = low + (range_size * cdf_int[symbol + 1]) // max_range
        low = low + (range_size * cdf_int[symbol]) // max_range

        # Renormalize
        while True:
            if high <= half:
                # Interval is in [0, 0.5) - output 0
                output_bit(0)
                low = low * 2
                high = high * 2
            elif low >= half:
                # Interval is in [0.5, 1) - output 1
                output_bit(1)
                low = (low - half) * 2
                high = (high - half) * 2
            elif low >= quarter and high <= 3 * quarter:
                # Interval straddles middle - expand and track
                pending_bits += 1
                low = (low - quarter) * 2
                high = (high - quarter) * 2
            else:
                break

    # Flush remaining bits
    pending_bits += 1
    if low < quarter:
        output_bit(0)
    else:
        output_bit(1)

    return bitstream, len(bitstream)


def arithmetic_decode(
    bitstream: list[int],
    cdf: Tensor,
    length: int,
    precision: int = 32,
) -> Tensor:
    """Decode an arithmetic-coded bitstream.

    Parameters
    ----------
    bitstream : list[int]
        Encoded bits from arithmetic_encode.
    cdf : Tensor
        Cumulative distribution function (same as used for encoding).
    length : int
        Number of symbols to decode.
    precision : int, default=32
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
    >>> bits, n = arithmetic_encode(symbols, cdf)
    >>> decoded = arithmetic_decode(bits, cdf, length=4)
    >>> torch.equal(decoded, symbols)
    True

    Notes
    -----
    - Not differentiable (discrete operation).

    See Also
    --------
    arithmetic_encode : Encode symbols with arithmetic coding.
    """
    if not isinstance(cdf, Tensor):
        raise TypeError(f"cdf must be a Tensor, got {type(cdf).__name__}")

    if length == 0:
        return torch.empty(0, dtype=torch.long)

    max_range = 1 << precision
    half = 1 << (precision - 1)
    quarter = 1 << (precision - 2)

    # Convert CDF to integers
    cdf_int = (cdf.float() * max_range).long().tolist()
    n_symbols = len(cdf_int) - 1

    # Initialize
    low = 0
    high = max_range

    # Read initial value from bitstream
    value = 0
    bit_idx = 0
    for _ in range(precision):
        value = value * 2
        if bit_idx < len(bitstream):
            value += bitstream[bit_idx]
            bit_idx += 1

    decoded = []

    for _ in range(length):
        # Find symbol whose interval contains the value
        range_size = high - low
        # Scaled value relative to current interval
        scaled = ((value - low + 1) * max_range - 1) // range_size

        # Binary search for symbol
        symbol = 0
        for s in range(n_symbols):
            if cdf_int[s + 1] > scaled:
                symbol = s
                break

        decoded.append(symbol)

        # Narrow interval
        high = low + (range_size * cdf_int[symbol + 1]) // max_range
        low = low + (range_size * cdf_int[symbol]) // max_range

        # Renormalize
        while True:
            if high <= half:
                low = low * 2
                high = high * 2
                value = value * 2
                if bit_idx < len(bitstream):
                    value += bitstream[bit_idx]
                    bit_idx += 1
            elif low >= half:
                low = (low - half) * 2
                high = (high - half) * 2
                value = (value - half) * 2
                if bit_idx < len(bitstream):
                    value += bitstream[bit_idx]
                    bit_idx += 1
            elif low >= quarter and high <= 3 * quarter:
                low = (low - quarter) * 2
                high = (high - quarter) * 2
                value = (value - quarter) * 2
                if bit_idx < len(bitstream):
                    value += bitstream[bit_idx]
                    bit_idx += 1
            else:
                break

    return torch.tensor(decoded, dtype=torch.long)
