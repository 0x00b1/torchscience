"""Range encoding and decoding."""

import torch
from torch import Tensor


def range_encode(
    symbols: Tensor,
    cdf: Tensor,
    precision: int = 16,
) -> tuple[list[int], int]:
    """Encode symbols using range coding.

    Range coding is a variant of arithmetic coding that uses integer
    arithmetic and outputs bytes, making it more practical for
    implementation while achieving similar compression ratios.

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
        Bit precision for frequency counts. Higher precision allows
        more accurate probability representation.

    Returns
    -------
    bytes_out : list[int]
        Encoded bytes (values 0-255).
    num_bytes : int
        Number of bytes in the output.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([0, 1, 0, 0])
    >>> cdf = torch.tensor([0.0, 0.7, 1.0])  # P(0)=0.7, P(1)=0.3
    >>> bytes_out, n = range_encode(symbols, cdf)

    Notes
    -----
    - Achieves compression rate approaching entropy H(X).
    - Not differentiable (discrete operation).
    - More efficient than arithmetic coding for practical implementations.

    See Also
    --------
    range_decode : Decode range-coded data.
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
    # Ensure last value is exactly total
    cdf_int[-1] = total

    symbols_list = symbols.tolist()
    n_symbols = len(cdf_int) - 1

    # Range coder constants
    TOP = 1 << 24
    BOTTOM = 1 << 16

    # State: range is [low, low + range)
    low = 0
    range_val = 0xFFFFFFFF

    bytes_out = []

    for symbol in symbols_list:
        if symbol < 0 or symbol >= n_symbols:
            raise ValueError(f"Symbol {symbol} out of range [0, {n_symbols})")

        # Get frequency bounds for this symbol
        freq_low = cdf_int[symbol]
        freq_high = cdf_int[symbol + 1]

        # Narrow the range
        # range = range / total * (freq_high - freq_low)
        # low = low + range / total * freq_low
        range_val = range_val // total
        low = low + range_val * freq_low
        range_val = range_val * (freq_high - freq_low)

        # Renormalize when range is too small
        while range_val < BOTTOM:
            # Check if low byte has overflowed
            if low < 0xFF000000:
                bytes_out.append((low >> 24) & 0xFF)
            elif low >= 0x01000000:
                bytes_out.append(((low >> 24) + 1) & 0xFF)
            else:
                # Handle carry propagation
                bytes_out.append((low >> 24) & 0xFF)

            low = (low << 8) & 0xFFFFFFFF
            range_val = range_val << 8

    # Flush remaining state
    for _ in range(4):
        bytes_out.append((low >> 24) & 0xFF)
        low = (low << 8) & 0xFFFFFFFF

    return bytes_out, len(bytes_out)


def range_decode(
    bytes_in: list[int],
    cdf: Tensor,
    length: int,
    precision: int = 16,
) -> Tensor:
    """Decode range-coded data.

    Parameters
    ----------
    bytes_in : list[int]
        Encoded bytes from range_encode.
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
    >>> bytes_out, n = range_encode(symbols, cdf)
    >>> decoded = range_decode(bytes_out, cdf, length=4)
    >>> torch.equal(decoded, symbols)
    True

    Notes
    -----
    - Not differentiable (discrete operation).

    See Also
    --------
    range_encode : Encode symbols with range coding.
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

    # Range coder constants
    BOTTOM = 1 << 16

    # Initialize state
    low = 0
    range_val = 0xFFFFFFFF

    # Read initial code value (4 bytes)
    code = 0
    byte_idx = 0
    for _ in range(4):
        if byte_idx < len(bytes_in):
            code = (code << 8) | bytes_in[byte_idx]
            byte_idx += 1
        else:
            code = code << 8

    decoded = []

    for _ in range(length):
        # Compute frequency from current code position
        range_val = range_val // total

        # The offset tells us which symbol interval we're in
        offset = (code - low) // range_val
        if offset >= total:
            offset = total - 1

        # Find symbol whose interval contains offset
        symbol = 0
        for s in range(n_symbols):
            if cdf_int[s + 1] > offset:
                symbol = s
                break

        decoded.append(symbol)

        # Update range exactly as encoder did
        freq_low = cdf_int[symbol]
        freq_high = cdf_int[symbol + 1]
        low = low + range_val * freq_low
        range_val = range_val * (freq_high - freq_low)

        # Renormalize
        while range_val < BOTTOM:
            low = (low << 8) & 0xFFFFFFFF
            range_val = range_val << 8
            # Read next byte into code
            if byte_idx < len(bytes_in):
                code = ((code << 8) | bytes_in[byte_idx]) & 0xFFFFFFFF
                byte_idx += 1
            else:
                code = (code << 8) & 0xFFFFFFFF

    return torch.tensor(decoded, dtype=torch.long)
