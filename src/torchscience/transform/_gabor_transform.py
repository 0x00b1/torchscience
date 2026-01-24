"""Gabor transform implementation."""

from typing import Literal

import torch
from torch import Tensor

from torchscience.pad import PaddingMode

from ._short_time_fourier_transform import short_time_fourier_transform


def gabor_transform(
    input: Tensor,
    *,
    sigma: float,
    n_fft: int = 256,
    hop_length: int | None = None,
    dim: int = -1,
    n: int | None = None,
    norm: Literal["forward", "backward", "ortho"] = "backward",
    padding: int | tuple[int, int] | None = None,
    padding_mode: PaddingMode = "constant",
    padding_value: float = 0.0,
    padding_order: int = 1,
    center: bool = True,
    out: Tensor | None = None,
) -> Tensor:
    r"""Compute the Gabor transform of a signal.

    The Gabor transform is a short-time Fourier transform (STFT) with a
    Gaussian window. It provides optimal time-frequency localization,
    achieving the theoretical minimum uncertainty in joint time-frequency
    representation.

    .. math::
        G_x(t, f) = \int_{-\infty}^{\infty} x(\tau) \cdot g(\tau - t) \cdot
                    e^{-2\pi i f \tau} d\tau

    where :math:`g(t)` is the Gaussian window:

    .. math::
        g(t) = e^{-\frac{t^2}{2\sigma^2}}

    The discrete implementation uses an STFT with a sampled Gaussian window.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. The transform is computed along ``dim``.
    sigma : float
        **Required.** Width of the Gaussian window as a fraction of ``n_fft``.
        The standard deviation of the Gaussian in samples is ``sigma * n_fft``.
        Smaller values give better time resolution but worse frequency
        resolution. Typical values are 0.05 to 0.3.
    n_fft : int, optional
        Size of Fourier transform (window length).
        Default: ``256``.
    hop_length : int, optional
        The distance between neighboring sliding window frames.
        Default: ``n_fft // 4``.
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
        The Gabor transform of the input. Shape is ``(..., n_fft//2 + 1, num_frames)``
        for real input, where ``...`` represents batch dimensions.
        Always complex-valued.

    Examples
    --------
    Basic usage:

    >>> x = torch.randn(1024)
    >>> X = gabor_transform(x, sigma=0.1, n_fft=256)
    >>> X.shape  # (n_fft//2 + 1, num_frames)
    torch.Size([129, 69])

    Different time-frequency tradeoffs:

    >>> X_narrow = gabor_transform(x, sigma=0.05)  # Better time resolution
    >>> X_wide = gabor_transform(x, sigma=0.2)    # Better frequency resolution

    Batched input:

    >>> x = torch.randn(4, 1024)
    >>> X = gabor_transform(x, sigma=0.1, n_fft=256, dim=-1)
    >>> X.shape
    torch.Size([4, 129, 69])

    Notes
    -----
    **Time-Frequency Tradeoff:**

    The ``sigma`` parameter controls the time-frequency tradeoff:

    - **Small sigma** (e.g., 0.05): Narrow Gaussian window gives good time
      resolution but poor frequency resolution. Good for detecting transients.
    - **Large sigma** (e.g., 0.3): Wide Gaussian window gives good frequency
      resolution but poor time resolution. Good for tonal analysis.

    The Gabor transform achieves the theoretical minimum uncertainty product
    in the time-frequency domain (Heisenberg-Gabor limit).

    **Relationship to STFT:**

    The Gabor transform is equivalent to an STFT with a Gaussian window.
    Internally, this function generates a Gaussian window of width
    ``sigma * n_fft`` samples and calls ``short_time_fourier_transform``.

    **Gaussian Window Generation:**

    The Gaussian window is computed as:

    .. math::
        w[n] = \exp\left(-\frac{1}{2}\left(\frac{n - (N-1)/2}{\sigma N}\right)^2\right)

    and normalized so that ``sum(w) * sqrt(N) = N`` for energy preservation.

    See Also
    --------
    inverse_gabor_transform : The inverse Gabor transform.
    short_time_fourier_transform : STFT with arbitrary window.
    """
    # Generate Gaussian window
    # sigma_samples is the standard deviation in samples
    sigma_samples = sigma * n_fft

    # Create centered time indices
    t = torch.arange(n_fft, dtype=input.dtype, device=input.device)
    t = t - (n_fft - 1) / 2.0  # Center the window

    # Compute Gaussian window
    window = torch.exp(-0.5 * (t / sigma_samples) ** 2)

    # Normalize: window.sum() * sqrt(n_fft) = n_fft
    # This gives window = window / window.sum() * sqrt(n_fft)
    window = window / window.sum() * (n_fft**0.5)

    # Call STFT with the Gaussian window
    return short_time_fourier_transform(
        input,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        dim=dim,
        n=n,
        norm=norm,
        padding=padding,
        padding_mode=padding_mode,
        padding_value=padding_value,
        padding_order=padding_order,
        center=center,
        out=out,
    )
