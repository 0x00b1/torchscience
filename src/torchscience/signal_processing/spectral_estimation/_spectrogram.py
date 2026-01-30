from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def spectrogram(
    x: Tensor,
    *,
    fs: float = 1.0,
    window: Optional[Tensor] = None,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute the spectrogram of a signal.

    Computes a time-frequency representation by computing the squared
    magnitude of the short-time Fourier transform (STFT) for each
    overlapping segment.

    Mathematical Definition
    -----------------------
    The spectrogram is defined as the squared magnitude of the STFT:

    .. math::

        S(f, t_k) = \frac{1}{f_s \cdot S}
            \left| \sum_{n=0}^{L-1} x[n + k \cdot H] \, w[n] \, e^{-j 2\pi f n / L} \right|^2

    where :math:`L` is the segment length, :math:`H = L - \text{noverlap}` is
    the hop size, and :math:`S = \sum w[n]^2`.

    Parameters
    ----------
    x : Tensor
        Input signal tensor of shape ``(..., N)``. The last dimension is
        the signal axis. Batch dimensions are fully supported.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    window : Tensor, optional
        Window tensor of shape ``(nperseg,)``. If ``None``, a Hann window
        is used.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Overlap between segments. Default is ``nperseg // 2``.
    scaling : {'density', 'spectrum'}, optional
        Selects between power spectral density ('density', V**2/Hz) and
        squared magnitude spectrum ('spectrum', V**2). Default is 'density'.

    Returns
    -------
    freqs : Tensor
        Frequency bins of shape ``(nperseg//2 + 1,)``.
    times : Tensor
        Time bins (segment centers) of shape ``(n_segments,)``.
    Sxx : Tensor
        Spectrogram of shape ``(..., nperseg//2 + 1, n_segments)``.

    Examples
    --------
    Compute the spectrogram of a chirp signal:

    >>> t = torch.arange(10000, dtype=torch.float64) / 1000.0
    >>> x = torch.sin(2 * torch.pi * (10 + 40 * t / 10) * t)
    >>> freqs, times, Sxx = spectrogram(x, fs=1000.0, nperseg=256)
    >>> Sxx.shape
    torch.Size([129, ...])

    References
    ----------
    - Allen, J.B. "Short term spectral analysis, synthesis, and modification
      by discrete Fourier transform." IEEE Trans. ASSP 25(3), 235-238 (1977).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if window is None:
        window = torch.hann_window(nperseg, dtype=x.dtype, device=x.device)

    scaling_int = 0 if scaling == "density" else 1

    return torch.ops.torchscience.spectrogram_psd(
        x, window, nperseg, noverlap, fs, scaling_int
    )
