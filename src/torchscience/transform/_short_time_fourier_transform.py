"""Short-time Fourier transform implementation."""

from typing import Literal

import torch
from torch import Tensor

from torchscience.pad import PaddingMode, pad

# Valid padding modes for validation
_VALID_PADDING_MODES = {
    "constant",
    "replicate",
    "reflect",
    "reflect_odd",
    "circular",
    "linear",
    "polynomial",
    "spline",
    "smooth",
}


def short_time_fourier_transform(
    input: Tensor,
    *,
    hop_length: int | None = None,
    n_fft: int | None = None,
    dim: int = -1,
    n: int | None = None,
    norm: Literal["forward", "backward", "ortho"] = "backward",
    padding: int | tuple[int, int] | None = None,
    padding_mode: PaddingMode = "constant",
    padding_value: float = 0.0,
    padding_order: int = 1,
    window: Tensor | None = None,
    center: bool = True,
    out: Tensor | None = None,
) -> Tensor:
    r"""Compute the short-time Fourier transform (STFT) of a signal.

    The STFT represents a signal in the time-frequency domain by computing
    the Fourier transform over short, overlapping segments of the signal.

    .. math::
        X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-2\pi i k n / N}

    where :math:`m` is the frame index, :math:`k` is the frequency bin,
    :math:`H` is the hop length, and :math:`w[n]` is the window function.

    This implementation wraps PyTorch's STFT with the unified transform contract,
    providing additional padding modes and flexible dimension handling.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. The transform is computed along ``dim``.
    hop_length : int, optional
        The distance between neighboring sliding window frames.
        Default: ``window.size(0) // 4``.
    n_fft : int, optional
        Size of Fourier transform. Must be >= ``window.size(0)``.
        Default: ``window.size(0)``.
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    n : int, optional
        Signal length. If given, the input will either be padded or
        truncated to this length before computing the transform.
        Default: ``None`` (use input size along ``dim``).
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization on forward transform.
        - ``'ortho'``: Normalize by 1/sqrt(n_fft).
        - ``'forward'``: Normalize by 1/n_fft.

        Default: ``'backward'``.
    padding : int or tuple of int, optional
        Explicit padding to apply before the transform (in addition to
        center padding if ``center=True``). Accepts:

        - ``int``: Same padding on both sides
        - ``(before, after)``: Asymmetric padding

        Default: ``None`` (no explicit padding beyond centering).
    padding_mode : str, optional
        Padding mode when padding is needed. One of:

        - ``'constant'``: Pad with ``padding_value`` (default 0).
        - ``'reflect'``: Reflect the signal at boundaries.
        - ``'reflect_odd'``: Antisymmetric reflection.
        - ``'replicate'``: Replicate edge values.
        - ``'circular'``: Wrap around (periodic extension).
        - ``'linear'``: Linear extrapolation from edge.
        - ``'polynomial'``: Polynomial extrapolation of degree ``padding_order``.
        - ``'spline'``: Cubic spline extrapolation.
        - ``'smooth'``: C1-continuous extension (matches value and derivative).

        Default: ``'constant'``.
    padding_value : float, optional
        Fill value for ``'constant'`` padding mode. Ignored for other modes.
        Default: ``0.0``.
    padding_order : int, optional
        Polynomial order for ``'polynomial'`` padding mode.
        Default: ``1`` (linear).
    window : Tensor
        **Required.** Window function to apply to each frame. Must be 1-D.
        Common windows: ``torch.hann_window``, ``torch.hamming_window``,
        ``torch.blackman_window``.
    center : bool, optional
        If ``True``, the signal is padded on both sides so that the first
        frame is centered on the first sample and the last frame is centered
        on the last sample.
        If ``False``, the first frame starts at the first sample.
        Default: ``True``.
    out : Tensor, optional
        Output tensor. Must have the correct shape and dtype (complex).
        Default: ``None`` (allocate new tensor).

    Returns
    -------
    Tensor
        The STFT of the input. Shape is ``(..., n_fft//2 + 1, num_frames)``
        for real input, where ``...`` represents batch dimensions.
        Always complex-valued.

    Raises
    ------
    ValueError
        If ``window`` is ``None`` (window is required for STFT).
        If ``padding_mode`` is not a valid mode.
        If ``window`` is not 1-D.

    Examples
    --------
    Basic usage:

    >>> x = torch.randn(1024)
    >>> window = torch.hann_window(256)
    >>> X = short_time_fourier_transform(x, window=window)
    >>> X.shape  # (n_fft//2 + 1, num_frames)
    torch.Size([129, 69])

    With custom hop length:

    >>> X = short_time_fourier_transform(x, window=window, hop_length=128)
    >>> X.shape
    torch.Size([129, 9])

    Batched input:

    >>> x = torch.randn(4, 1024)
    >>> X = short_time_fourier_transform(x, window=window, dim=-1)
    >>> X.shape
    torch.Size([4, 129, 69])

    Notes
    -----
    **Window Requirement:**

    Unlike some other transforms in this module, the STFT requires a window
    function. This is because the window is essential for proper time-frequency
    analysis and reconstruction. Use ``torch.hann_window``, ``torch.hamming_window``,
    or other window functions.

    **Relationship to torch.stft:**

    This function wraps ``torch.stft`` with the unified transform contract,
    adding support for:

    - All 9 padding modes from ``torchscience.pad``
    - Flexible dimension handling via ``dim`` parameter
    - Signal length control via ``n`` parameter
    - Consistent parameter naming with other transforms

    **Frame Count:**

    For a signal of length L with center=True:
        num_frames = (L + n_fft) // hop_length

    For center=False:
        num_frames = (L - n_fft) // hop_length + 1

    **Gradient Computation:**

    Gradients are computed analytically via torch.stft's autograd support.

    See Also
    --------
    inverse_short_time_fourier_transform : The inverse STFT.
    torch.stft : PyTorch's STFT implementation.
    fourier_transform : Standard Fourier transform.
    """
    # Validate window is provided
    if window is None:
        raise ValueError(
            "window is required for short_time_fourier_transform. "
            "Use torch.hann_window, torch.hamming_window, or similar."
        )

    # Validate window is 1-D
    if window.ndim != 1:
        raise ValueError(
            f"window must be 1-D, got {window.ndim}-D tensor with shape {window.shape}"
        )

    # Validate padding_mode
    if padding_mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(_VALID_PADDING_MODES)}, "
            f"got '{padding_mode}'"
        )

    win_length = window.size(0)

    # Set defaults based on window size
    if n_fft is None:
        n_fft = win_length
    if hop_length is None:
        hop_length = win_length // 4

    # Validate n_fft >= win_length
    if n_fft < win_length:
        raise ValueError(
            f"n_fft ({n_fft}) must be >= window length ({win_length})"
        )

    # Normalize dimension
    ndim = input.ndim
    normalized_dim = dim if dim >= 0 else dim + ndim

    # Move transform dimension to last position for torch.stft
    if normalized_dim != ndim - 1:
        x = input.movedim(normalized_dim, -1)
    else:
        x = input

    # Handle n parameter (signal length control)
    current_size = x.shape[-1]
    if n is not None:
        if n > current_size:
            # Pad to reach target size
            pad_amount = n - current_size
            x = pad(
                x,
                (0, pad_amount),
                mode=padding_mode,
                value=padding_value,
                dim=-1,
                order=padding_order,
            )
        elif n < current_size:
            # Truncate
            x = x.narrow(-1, 0, n)

    # Apply explicit padding if specified
    if padding is not None:
        if isinstance(padding, int):
            pad_spec = (padding, padding)
        else:
            pad_spec = padding

        x = pad(
            x,
            pad_spec,
            mode=padding_mode,
            value=padding_value,
            dim=-1,
            order=padding_order,
        )

    # Map norm to torch.stft's normalized parameter
    # torch.stft uses normalized: bool | None
    # backward -> False, ortho -> True, forward -> requires manual scaling
    if norm == "backward":
        normalized = False
    elif norm == "ortho":
        normalized = True
    else:  # forward
        normalized = False  # We'll scale manually after

    # torch.stft only accepts 1D or 2D tensors.
    # For higher dimensional inputs, we need to flatten batch dimensions,
    # apply stft, then unflatten.
    original_shape = x.shape
    signal_length = original_shape[-1]

    if x.ndim == 1:
        # 1D input - use directly
        batch_shape = ()
        x_2d = x.unsqueeze(0)  # Add batch dim for uniform handling
        had_batch = False
    elif x.ndim == 2:
        # 2D input - use directly
        batch_shape = original_shape[:-1]
        x_2d = x
        had_batch = True
    else:
        # Higher dimensional - flatten all batch dims into one
        batch_shape = original_shape[:-1]
        batch_numel = 1
        for s in batch_shape:
            batch_numel *= s
        x_2d = x.reshape(batch_numel, signal_length)
        had_batch = True

    # Determine pad_mode for torch.stft
    # torch.stft supports: 'constant', 'reflect', 'replicate', 'circular'
    # For other modes, we do centering ourselves
    torch_pad_modes = {"constant", "reflect", "replicate", "circular"}

    if padding_mode in torch_pad_modes and center:
        # Let torch.stft handle the centering
        result_2d = torch.stft(
            x_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode=padding_mode,
            normalized=normalized,
            onesided=True,
            return_complex=True,
        )
    elif center:
        # Apply centering padding ourselves using the advanced padding mode
        center_pad = n_fft // 2
        x_2d = pad(
            x_2d,
            (center_pad, center_pad),
            mode=padding_mode,
            value=padding_value,
            dim=-1,
            order=padding_order,
        )
        result_2d = torch.stft(
            x_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            normalized=normalized,
            onesided=True,
            return_complex=True,
        )
    else:
        # No centering
        result_2d = torch.stft(
            x_2d,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            normalized=normalized,
            onesided=True,
            return_complex=True,
        )

    # Apply forward normalization manually if needed
    if norm == "forward":
        result_2d = result_2d / n_fft

    # result_2d has shape (batch, freq, frames)
    freq_bins = result_2d.shape[-2]
    num_frames = result_2d.shape[-1]

    # Restore batch dimensions
    # result_2d has shape (batch, freq, frames)
    # After unflatten, we want (..., freq, frames) where ... are batch dims
    # Note: the batch_shape here is the shape AFTER moving the signal dim to the end
    if not had_batch:
        # Remove the batch dim we added
        result = result_2d.squeeze(0)  # (freq, frames)
    elif len(batch_shape) == 1:
        # Already correct shape: (batch, freq, frames)
        result = result_2d
    else:
        # Unflatten batch dimensions
        result = result_2d.reshape(*batch_shape, freq_bins, num_frames)

    # The result now has shape (..., freq, frames) where ... matches batch_shape
    # This is the expected output format - we don't move dimensions back
    # because STFT transforms one dimension into two (freq, frames)

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result
