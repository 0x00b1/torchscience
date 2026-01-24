"""Fourier transform implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Padding mode mapping
_PADDING_MODES = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}

# Normalization mode mapping
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
    "forward": 2,
}


def fourier_transform(
    input: Tensor,
    *,
    n: Optional[int] = None,
    dim: int = -1,
    padding_mode: Literal[
        "constant", "reflect", "replicate", "circular"
    ] = "constant",
    padding_value: float = 0.0,
    window: Optional[Tensor] = None,
    norm: Optional[Literal["forward", "backward", "ortho"]] = None,
) -> Tensor:
    r"""Compute the discrete Fourier transform of a signal.

    The Fourier transform is defined as:

    .. math::
        X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi i k n / N}

    This implementation wraps PyTorch's FFT with additional support for
    padding and windowing.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Can be real or complex.
    n : int, optional
        Signal length. If given, the input will either be padded or
        truncated to this length before computing the transform.
        Default: ``None`` (use input size along ``dim``).
    dim : int, optional
        The dimension along which to compute the transform.
        Default: ``-1`` (last dimension).
    padding_mode : str, optional
        Padding mode when ``n`` is larger than input size. One of:

        - ``'constant'``: Pad with ``padding_value`` (default 0).
        - ``'reflect'``: Reflect the signal at boundaries.
        - ``'replicate'``: Replicate edge values.
        - ``'circular'``: Wrap around (periodic extension).

        Default: ``'constant'``.
    padding_value : float, optional
        Fill value for ``'constant'`` padding mode. Ignored for other modes.
        Default: ``0.0``.
    window : Tensor, optional
        Window function to apply before the transform. Must be 1-D with size
        matching the (possibly padded) signal length along ``dim``.
        Use window functions from ``torch`` (e.g., ``torch.hann_window``).
        Default: ``None`` (no windowing).
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization on forward, divide by n on inverse.
        - ``'ortho'``: Normalize by 1/sqrt(n) on both forward and inverse.
        - ``'forward'``: Divide by n on forward, no normalization on inverse.

        Default: ``None`` (equivalent to ``'backward'``).

    Returns
    -------
    Tensor
        The Fourier transform of the input. Always complex-valued.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1., 2., 3., 4.])
    >>> X = fourier_transform(x)
    >>> X.shape
    torch.Size([4])
    >>> X.dtype
    torch.complex64

    Compare with torch.fft.fft:

    >>> torch.allclose(X, torch.fft.fft(x))
    True

    With padding to get more frequency resolution:

    >>> x = torch.randn(64)
    >>> X = fourier_transform(x, n=128)
    >>> X.shape
    torch.Size([128])

    With a window function:

    >>> x = torch.randn(100)
    >>> window = torch.hann_window(100)
    >>> X = fourier_transform(x, window=window)

    Notes
    -----
    **Windowing:**

    Applying a window function before the transform reduces spectral leakage.
    Common windows include:

    - ``torch.hann_window``: Good general-purpose window
    - ``torch.hamming_window``: Similar to Hann
    - ``torch.blackman_window``: Better sidelobe suppression

    **Normalization:**

    - ``'backward'`` (default): The forward transform is unnormalized, and
      the inverse is normalized by 1/n. This is the most common convention.
    - ``'ortho'``: Both transforms are normalized by 1/sqrt(n), making the
      transform unitary.
    - ``'forward'``: The forward transform is normalized by 1/n, and the
      inverse is unnormalized.

    **Implementation:**

    Uses PyTorch's FFT backend (cuFFT on CUDA, MKL/FFTW on CPU).

    **Gradient Computation:**

    Gradients are computed analytically. The Fourier transform is a linear
    operator, so:

    .. math::
        \frac{\partial L}{\partial x} = \text{IFFT}\left[\frac{\partial L}{\partial X}\right]

    (with appropriate normalization adjustments). Second-order gradients are
    also supported.

    See Also
    --------
    inverse_fourier_transform : The inverse Fourier transform.
    torch.fft.fft : PyTorch's FFT implementation.
    """
    if padding_mode not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {list(_PADDING_MODES.keys())}, "
            f"got '{padding_mode}'"
        )

    norm_int = _NORM_MODES.get(norm, 0) if norm is not None else 0

    return torch.ops.torchscience.fourier_transform(
        input,
        n if n is not None else -1,
        dim,
        _PADDING_MODES[padding_mode],
        padding_value,
        window,
        norm_int,
    )
