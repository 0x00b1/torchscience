"""Inverse Gabor transform implementation."""

from typing import Literal

import torch
from torch import Tensor

from torchscience.pad import PaddingMode

from ._inverse_short_time_fourier_transform import (
    inverse_short_time_fourier_transform,
)


def inverse_gabor_transform(
    input: Tensor,
    *,
    sigma: float,
    n_fft: int = 256,
    hop_length: int | None = None,
    dim: int = -1,
    n: int | None = None,
    length: int | None = None,
    norm: Literal["forward", "backward", "ortho"] = "backward",
    padding: int | tuple[int, int] | None = None,
    padding_mode: PaddingMode = "constant",
    padding_value: float = 0.0,
    padding_order: int = 1,
    center: bool = True,
    out: Tensor | None = None,
) -> Tensor:
    r"""Compute the inverse Gabor transform of a signal.

    The inverse Gabor transform reconstructs a time-domain signal from its
    Gabor transform representation using the overlap-add method. It uses the
    same Gaussian window that was used in the forward Gabor transform.

    .. math::
        x[n] = \frac{\sum_m G_x[m, k] \cdot g[n - mH] \cdot e^{2\pi i k n / N}}
                    {\sum_m g^2[n - mH]}

    where :math:`m` is the frame index, :math:`k` is the frequency bin,
    :math:`H` is the hop length, and :math:`g[n]` is the Gaussian window.

    This implementation reconstructs the same Gaussian window used in the
    forward transform and calls ``inverse_short_time_fourier_transform``.

    Parameters
    ----------
    input : Tensor
        Input tensor containing the Gabor transform representation. Must be
        complex-valued. Expected shape is ``(..., freq_bins, num_frames)``
        where ``...`` are batch dimensions.
    sigma : float
        **Required.** Width of the Gaussian window as a fraction of ``n_fft``.
        Must match the ``sigma`` used in the forward Gabor transform.
        The standard deviation of the Gaussian in samples is ``sigma * n_fft``.
    n_fft : int, optional
        Size of Fourier transform used in the forward Gabor transform.
        Must match the ``n_fft`` used in the forward transform.
        Default: ``256``.
    hop_length : int, optional
        The distance between neighboring sliding window frames.
        Must match the ``hop_length`` used in the forward transform.
        Default: ``n_fft // 4``.
    dim : int, optional
        The dimension along which the Gabor transform was computed.
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

        Should match the normalization used in the forward Gabor transform.
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
    center : bool, optional
        If ``True``, the signal was padded on both sides in the forward
        transform, and this padding is removed.
        If ``False``, no centering adjustment is made.
        Should match the ``center`` parameter used in the forward Gabor transform.
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

    Examples
    --------
    Round-trip reconstruction:

    >>> x = torch.randn(1024)
    >>> G = gabor_transform(x, sigma=0.1, n_fft=256)
    >>> x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=256, length=1024)
    >>> torch.allclose(x, x_rec, atol=1e-4)
    True

    Batched reconstruction:

    >>> x = torch.randn(4, 1024)
    >>> G = gabor_transform(x, sigma=0.1, n_fft=256, dim=-1)
    >>> x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=256, length=1024)
    >>> x_rec.shape
    torch.Size([4, 1024])

    Different time-frequency tradeoffs:

    >>> x = torch.randn(512)
    >>> # Must use same sigma in forward and inverse
    >>> G_narrow = gabor_transform(x, sigma=0.05, n_fft=128)
    >>> x_narrow = inverse_gabor_transform(G_narrow, sigma=0.05, n_fft=128, length=512)
    >>> G_wide = gabor_transform(x, sigma=0.2, n_fft=128)
    >>> x_wide = inverse_gabor_transform(G_wide, sigma=0.2, n_fft=128, length=512)

    Notes
    -----
    **Sigma Matching:**

    The ``sigma`` parameter **must** match the ``sigma`` used in the forward
    Gabor transform. Using a different ``sigma`` will result in incorrect
    reconstruction because the Gaussian window will not match.

    **Perfect Reconstruction:**

    Perfect reconstruction is achieved when:

    1. The same ``sigma`` is used in forward and inverse transforms.
    2. The same ``n_fft``, ``hop_length``, ``center``, and ``norm``
       parameters are used.
    3. The Gaussian window satisfies the COLA (Constant Overlap-Add)
       condition, which is generally true for the default hop_length.

    **Gaussian Window Generation:**

    The Gaussian window is computed identically to ``gabor_transform``:

    .. math::
        w[n] = \exp\left(-\frac{1}{2}\left(\frac{n - (N-1)/2}{\sigma N}\right)^2\right)

    and normalized so that ``sum(w) * sqrt(N) = N`` for energy preservation.

    **Gradient Computation:**

    Gradients are computed analytically via torch.istft's autograd support.

    See Also
    --------
    gabor_transform : The forward Gabor transform.
    inverse_short_time_fourier_transform : ISTFT with arbitrary window.
    """
    # Determine the dtype for the window from the input
    # For complex input, use the real dtype component
    if input.is_complex():
        if input.dtype == torch.complex64:
            window_dtype = torch.float32
        elif input.dtype == torch.complex128:
            window_dtype = torch.float64
        else:
            window_dtype = torch.float32
    else:
        window_dtype = input.dtype

    # Generate Gaussian window (must match forward transform exactly)
    # sigma_samples is the standard deviation in samples
    sigma_samples = sigma * n_fft

    # Create centered time indices
    t = torch.arange(n_fft, dtype=window_dtype, device=input.device)
    t = t - (n_fft - 1) / 2.0  # Center the window

    # Compute Gaussian window
    window = torch.exp(-0.5 * (t / sigma_samples) ** 2)

    # Normalize: window.sum() * sqrt(n_fft) = n_fft
    # This gives window = window / window.sum() * sqrt(n_fft)
    window = window / window.sum() * (n_fft**0.5)

    # Call ISTFT with the Gaussian window
    return inverse_short_time_fourier_transform(
        input,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        dim=dim,
        n=n,
        length=length,
        norm=norm,
        padding=padding,
        padding_mode=padding_mode,
        padding_value=padding_value,
        padding_order=padding_order,
        center=center,
        out=out,
    )
