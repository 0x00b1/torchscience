"""FFT-based FIR filter implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def fftfilt(
    b: Tensor,
    x: Tensor,
    axis: int = -1,
) -> Tensor:
    """
    Apply an FIR filter using FFT-based convolution.

    This function implements efficient FIR filtering using FFT convolution.
    The result is equivalent to scipy.signal.fftconvolve(x, b, mode='same'),
    producing output the same shape as the input signal x.

    Parameters
    ----------
    b : Tensor
        FIR filter coefficients, shape (num_taps,).
    x : Tensor
        Input signal. Can be batched with arbitrary leading dimensions.
    axis : int, optional
        Axis along which to filter. Default is -1 (last axis).

    Returns
    -------
    y : Tensor
        Filtered signal, same shape as x.

    Notes
    -----
    This function uses FFT-based convolution which is efficient for
    long signals. The computational complexity is O(N log N) where N
    is the signal length, compared to O(N * M) for direct convolution
    where M is the filter length.

    The output is equivalent to:
        scipy.signal.fftconvolve(x, b, mode='same')

    Fully differentiable with respect to both b and x.

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter import firwin, fftfilt
    >>> b = firwin(51, 0.3)  # Lowpass filter
    >>> x = torch.randn(1000)  # Input signal
    >>> y = fftfilt(b, x)  # Filtered signal
    >>> y.shape
    torch.Size([1000])
    """
    # Move axis to last position for processing
    x = torch.moveaxis(x, axis, -1)
    original_shape = x.shape
    n_signal = x.shape[-1]
    n_filter = b.shape[0]

    # Flatten batch dimensions
    x_flat = x.reshape(-1, n_signal)

    # Compute FFT size (next power of 2 for efficiency)
    n_conv = n_signal + n_filter - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2

    # Determine if we need complex FFT
    use_complex = x.is_complex() or b.is_complex()

    if use_complex:
        # Use complex FFT for complex inputs
        B = torch.fft.fft(b, n=n_fft)
        X = torch.fft.fft(x_flat, n=n_fft)
        Y = B.unsqueeze(0) * X
        y_full = torch.fft.ifft(Y, n=n_fft)
    else:
        # Use real FFT for real inputs (more efficient)
        B = torch.fft.rfft(b, n=n_fft)
        X = torch.fft.rfft(x_flat, n=n_fft)
        Y = B.unsqueeze(0) * X
        y_full = torch.fft.irfft(Y, n=n_fft)

    # Extract 'same' size output (centered on the signal)
    # For mode='same' equivalent to fftconvolve(x, b, mode='same'),
    # the start offset is always (n_filter - 1) // 2 regardless of
    # which array is longer.
    start = (n_filter - 1) // 2
    y_flat = y_full[:, start : start + n_signal]

    # Restore original shape
    y = y_flat.reshape(original_shape)
    y = torch.moveaxis(y, -1, axis)

    return y
