from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.window_function import hann_window


def welch(
    x: Tensor,
    *,
    fs: float = 1.0,
    window: Optional[Tensor] = None,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[Tensor, Tensor]:
    r"""
    Power spectral density estimate using Welch's method.

    Welch's method divides the signal into overlapping segments, computes a
    modified periodogram for each segment, and averages the periodograms to
    reduce variance.

    Mathematical Definition
    -----------------------
    Given a signal :math:`x[n]` divided into :math:`K` overlapping segments
    :math:`x_k[n]` of length :math:`L` with window :math:`w[n]`:

    .. math::

        P_{\text{welch}}(f) = \frac{1}{K} \sum_{k=0}^{K-1}
            \frac{1}{f_s \cdot S}
            \left| \sum_{n=0}^{L-1} x_k[n] \, w[n] \, e^{-j 2\pi f n / L} \right|^2

    where :math:`S = \sum_{n=0}^{L-1} w[n]^2`.

    Parameters
    ----------
    x : Tensor
        Input signal tensor of shape ``(..., N)``. The last dimension is the
        signal axis. Batch dimensions are fully supported.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    window : Tensor, optional
        Window tensor of shape ``(nperseg,)``. If ``None``, a Hann window
        is used.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int, optional
        Number of overlapping samples between segments. Default is
        ``nperseg // 2``.
    scaling : {'density', 'spectrum'}, optional
        Selects between power spectral density ('density', V**2/Hz) and
        squared magnitude spectrum ('spectrum', V**2). Default is 'density'.

    Returns
    -------
    freqs : Tensor
        Frequency bins of shape ``(nperseg//2 + 1,)``.
    Pxx : Tensor
        Power spectral density of shape ``(..., nperseg//2 + 1)``.

    Examples
    --------
    Estimate PSD of a noisy sine wave:

    >>> torch.manual_seed(0)
    >>> t = torch.arange(10000, dtype=torch.float64) / 1000.0
    >>> x = torch.sin(2 * torch.pi * 50 * t) + 0.5 * torch.randn_like(t)
    >>> freqs, psd = welch(x, fs=1000.0, nperseg=256)
    >>> freqs[psd.argmax()]  # Peak near 50 Hz
    tensor(...)

    References
    ----------
    - Welch, P.D. "The use of fast Fourier transform for the estimation of
      power spectra." IEEE Trans. Audio Electroacoustics 15(2), 70-73 (1967).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if window is None:
        window = hann_window(nperseg, dtype=x.dtype, device=x.device)

    if scaling == "density":
        scaling_int = 0
    else:
        scaling_int = 1

    return torch.ops.torchscience.welch(
        x,
        window,
        nperseg,
        noverlap,
        fs,
        scaling_int,
    )
