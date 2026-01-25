"""Run-length encoding and decoding."""

import torch
from torch import Tensor


def run_length_encode(
    symbols: Tensor,
) -> tuple[Tensor, Tensor]:
    """Encode a sequence using run-length encoding.

    Run-length encoding (RLE) compresses sequences by storing consecutive
    runs of identical values as (value, count) pairs.

    Parameters
    ----------
    symbols : Tensor
        Input sequence to encode. Shape: ``(n,)`` where n is sequence length.
        Must be 1-dimensional.

    Returns
    -------
    values : Tensor
        Unique values at the start of each run. Shape: ``(n_runs,)``.
    run_lengths : Tensor
        Length of each run. Shape: ``(n_runs,)``.

    Examples
    --------
    >>> import torch
    >>> symbols = torch.tensor([1, 1, 1, 2, 2, 3])
    >>> values, lengths = run_length_encode(symbols)
    >>> values
    tensor([1, 2, 3])
    >>> lengths
    tensor([3, 2, 1])

    >>> # Single repeated value
    >>> symbols = torch.tensor([5, 5, 5, 5])
    >>> values, lengths = run_length_encode(symbols)
    >>> values
    tensor([5])
    >>> lengths
    tensor([4])

    Notes
    -----
    - RLE is most effective for sequences with many consecutive repetitions.
    - For alternating sequences, RLE may increase size.
    - Not differentiable (discrete operation).

    See Also
    --------
    run_length_decode : Decode run-length encoded data.
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
        return (
            torch.empty(0, dtype=symbols.dtype, device=symbols.device),
            torch.empty(0, dtype=torch.long, device=symbols.device),
        )

    # Find positions where value changes
    # diff[i] = True if symbols[i] != symbols[i+1]
    changes = symbols[:-1] != symbols[1:]

    # Indices where runs start (first element always starts a run)
    run_starts = torch.cat(
        [
            torch.tensor([0], device=symbols.device),
            torch.where(changes)[0] + 1,
        ]
    )

    # Values at run starts
    values = symbols[run_starts]

    # Run lengths: difference between consecutive start positions
    run_ends = torch.cat(
        [run_starts[1:], torch.tensor([len(symbols)], device=symbols.device)]
    )
    run_lengths = run_ends - run_starts

    return values, run_lengths


def run_length_decode(
    values: Tensor,
    run_lengths: Tensor,
) -> Tensor:
    """Decode run-length encoded data.

    Parameters
    ----------
    values : Tensor
        Values at the start of each run. Shape: ``(n_runs,)``.
    run_lengths : Tensor
        Length of each run. Shape: ``(n_runs,)``.

    Returns
    -------
    Tensor
        Decoded sequence. Shape: ``(sum(run_lengths),)``.

    Examples
    --------
    >>> import torch
    >>> values = torch.tensor([1, 2, 3])
    >>> lengths = torch.tensor([3, 2, 1])
    >>> run_length_decode(values, lengths)
    tensor([1, 1, 1, 2, 2, 3])

    Notes
    -----
    - Not differentiable (discrete operation).

    See Also
    --------
    run_length_encode : Encode data with run-length encoding.
    """
    if not isinstance(values, Tensor):
        raise TypeError(
            f"values must be a Tensor, got {type(values).__name__}"
        )
    if not isinstance(run_lengths, Tensor):
        raise TypeError(
            f"run_lengths must be a Tensor, got {type(run_lengths).__name__}"
        )

    if values.dim() != 1:
        raise ValueError(f"values must be 1-dimensional, got {values.dim()}D")
    if run_lengths.dim() != 1:
        raise ValueError(
            f"run_lengths must be 1-dimensional, got {run_lengths.dim()}D"
        )

    if len(values) != len(run_lengths):
        raise ValueError(
            f"values and run_lengths must have same length, "
            f"got {len(values)} and {len(run_lengths)}"
        )

    if len(values) == 0:
        return torch.empty(0, dtype=values.dtype, device=values.device)

    # Use repeat_interleave to expand values by their run lengths
    return torch.repeat_interleave(values, run_lengths.long())
