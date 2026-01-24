"""Inverse short-time Fourier transform implementation."""

from typing import Literal

import torch
from torch import Tensor

from torchscience.pad import PaddingMode

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


def inverse_short_time_fourier_transform(
    input: Tensor,
    *,
    hop_length: int | None = None,
    n_fft: int | None = None,
    dim: int = -1,
    n: int | None = None,
    length: int | None = None,
    norm: Literal["forward", "backward", "ortho"] = "backward",
    padding: int | tuple[int, int] | None = None,
    padding_mode: PaddingMode = "constant",
    padding_value: float = 0.0,
    padding_order: int = 1,
    window: Tensor | None = None,
    center: bool = True,
    out: Tensor | None = None,
) -> Tensor:
    r"""Compute the inverse short-time Fourier transform (ISTFT) of a signal.

    The ISTFT reconstructs a time-domain signal from its STFT representation
    using the overlap-add method.

    .. math::
        x[n] = \frac{\sum_m X[m, k] \cdot w[n - mH] \cdot e^{2\pi i k n / N}}
                    {\sum_m w^2[n - mH]}

    where :math:`m` is the frame index, :math:`k` is the frequency bin,
    :math:`H` is the hop length, and :math:`w[n]` is the window function.

    This implementation wraps PyTorch's ISTFT with the unified transform contract,
    providing consistent parameter naming and flexible dimension handling.

    Parameters
    ----------
    input : Tensor
        Input tensor containing the STFT representation. Must be complex-valued.
        Expected shape is ``(..., freq_bins, num_frames)`` where ``...`` are
        batch dimensions.
    hop_length : int, optional
        The distance between neighboring sliding window frames.
        Default: ``window.size(0) // 4``.
    n_fft : int, optional
        Size of Fourier transform used in the forward STFT.
        Default: ``2 * (input.shape[-2] - 1)`` (inferred from one-sided input).
    dim : int, optional
        The dimension along which the STFT was computed.
        Currently only supports ``-1`` (last dimension).
        Default: ``-1``.
    n : int, optional
        Target output signal length. Alias for ``length``.
        Default: ``None``.
    length : int, optional
        Target output signal length. Takes precedence over ``n`` if both provided.
        Default: ``None`` (infer from input).
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization on forward transform.
        - ``'ortho'``: Normalize by 1/sqrt(n_fft).
        - ``'forward'``: Normalize by 1/n_fft.

        Should match the normalization used in the forward STFT.
        Default: ``'backward'``.
    padding : int or tuple of int, optional
        Reserved for future use. Not currently implemented.
        Default: ``None``.
    padding_mode : str, optional
        Reserved for future use. Not currently implemented.
        Default: ``'constant'``.
    padding_value : float, optional
        Reserved for future use. Not currently implemented.
        Default: ``0.0``.
    padding_order : int, optional
        Reserved for future use. Not currently implemented.
        Default: ``1``.
    window : Tensor
        **Required.** Window function that was used in the forward STFT.
        Must be 1-D and match the window used in the forward transform.
    center : bool, optional
        If ``True``, the signal was padded on both sides in the forward
        transform, and this padding is removed.
        If ``False``, no centering adjustment is made.
        Should match the ``center`` parameter used in the forward STFT.
        Default: ``True``.
    out : Tensor, optional
        Output tensor. Must have the correct shape and dtype (real).
        Default: ``None`` (allocate new tensor).

    Returns
    -------
    Tensor
        The reconstructed time-domain signal. Shape is ``(..., signal_length)``
        where ``...`` represents batch dimensions.
        Always real-valued for real-valued input signals.

    Raises
    ------
    ValueError
        If ``window`` is ``None`` (window is required for ISTFT).
        If ``window`` is not 1-D.
        If ``padding_mode`` is not a valid mode.

    Examples
    --------
    Round-trip reconstruction:

    >>> x = torch.randn(1024)
    >>> window = torch.hann_window(256)
    >>> S = short_time_fourier_transform(x, window=window)
    >>> x_rec = inverse_short_time_fourier_transform(S, window=window, length=1024)
    >>> torch.allclose(x, x_rec, atol=1e-5)
    True

    Batched reconstruction:

    >>> x = torch.randn(4, 1024)
    >>> window = torch.hann_window(256)
    >>> S = short_time_fourier_transform(x, window=window, dim=-1)
    >>> x_rec = inverse_short_time_fourier_transform(S, window=window, length=1024)
    >>> x_rec.shape
    torch.Size([4, 1024])

    Notes
    -----
    **Window Requirement:**

    The ISTFT requires the same window function that was used in the forward
    STFT. Using a different window will result in incorrect reconstruction.

    **Perfect Reconstruction:**

    Perfect reconstruction is achieved when the COLA (Constant Overlap-Add)
    condition is satisfied:

    .. math::
        \sum_m w^2[n - mH] = \text{const}

    Common window/hop combinations that satisfy COLA:
    - Hann window with hop = window_length // 4
    - Hamming window with hop = window_length // 2

    **Normalization:**

    The ``norm`` parameter should match the normalization used in the forward
    STFT for proper reconstruction.

    **Gradient Computation:**

    Gradients are computed analytically via torch.istft's autograd support.

    See Also
    --------
    short_time_fourier_transform : The forward STFT.
    torch.istft : PyTorch's ISTFT implementation.
    inverse_fourier_transform : Standard inverse Fourier transform.
    """
    # Validate window is provided
    if window is None:
        raise ValueError(
            "window is required for inverse_short_time_fourier_transform. "
            "Must match the window used in the forward STFT."
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

    # Resolve length parameter (length takes precedence over n)
    target_length: int | None = length if length is not None else n

    # Infer n_fft from input shape if not provided
    # Input shape: (..., freq_bins, num_frames)
    # For one-sided STFT: freq_bins = n_fft // 2 + 1
    # So: n_fft = 2 * (freq_bins - 1)
    freq_bins = input.shape[-2]
    if n_fft is None:
        n_fft = 2 * (freq_bins - 1)

    # Set default hop_length based on window size
    if hop_length is None:
        hop_length = win_length // 4

    # Map norm to torch.istft's normalized parameter
    if norm == "backward":
        normalized = False
    elif norm == "ortho":
        normalized = True
    else:  # forward
        normalized = False  # We'll scale manually

    # torch.istft expects input shape: (batch, freq, frames) or (freq, frames)
    # Our input has shape: (..., freq, frames)
    # We need to flatten batch dimensions similar to STFT

    original_shape = input.shape
    freq_bins = original_shape[-2]
    num_frames = original_shape[-1]

    if input.ndim == 2:
        # 2D input (freq, frames) - no batch dimension
        batch_shape: tuple[int, ...] = ()
        x_2d = input.unsqueeze(0)  # Add batch dim
        had_batch = False
    elif input.ndim == 3:
        # 3D input (batch, freq, frames) - use directly
        batch_shape = original_shape[:-2]
        x_2d = input
        had_batch = True
    else:
        # Higher dimensional - flatten all batch dims into one
        batch_shape = original_shape[:-2]
        batch_numel = 1
        for s in batch_shape:
            batch_numel *= s
        x_2d = input.reshape(batch_numel, freq_bins, num_frames)
        had_batch = True

    # For forward normalization, we need to scale back before ISTFT
    if norm == "forward":
        x_2d = x_2d * n_fft

    # Call torch.istft
    result_2d = torch.istft(
        x_2d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=True,
        length=target_length,
        return_complex=False,
    )

    # result_2d has shape (batch, signal_length) or (signal_length,)
    signal_length = result_2d.shape[-1]

    # Restore batch dimensions
    if not had_batch:
        # Remove the batch dim we added
        result = result_2d.squeeze(0)
    elif len(batch_shape) == 1:
        # Already correct shape: (batch, signal_length)
        result = result_2d
    else:
        # Unflatten batch dimensions
        result = result_2d.reshape(*batch_shape, signal_length)

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result
