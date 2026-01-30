from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
from torchscience.window_function import hann_window


def cross_spectral_density(
    x: Tensor,
    y: Tensor,
    *,
    fs: float = 1.0,
    window: Optional[Tensor] = None,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[Tensor, Tensor]:
    r"""
    Cross spectral density estimate using Welch's method.

    Estimates the cross power spectral density of two real-valued signals
    by averaging the cross-periodograms of overlapping segments.

    Mathematical Definition
    -----------------------
    For signals :math:`x[n]` and :math:`y[n]` divided into :math:`K` segments:

    .. math::

        P_{xy}(f) = \frac{1}{K} \sum_{k=0}^{K-1}
            \frac{1}{f_s \cdot S}
            X_k^*(f) \, Y_k(f)

    where :math:`X_k(f)` and :math:`Y_k(f)` are the FFTs of the windowed
    segments and :math:`^*` denotes complex conjugate.

    Parameters
    ----------
    x : Tensor
        First input signal of shape ``(..., N)``.
    y : Tensor
        Second input signal of shape ``(..., N)``. Must be broadcastable
        with ``x``.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    window : Tensor, optional
        Window tensor of shape ``(nperseg,)``. If ``None``, a Hann window
        is used.
    nperseg : int, optional
        Segment length. Default is 256.
    noverlap : int, optional
        Overlap between segments. Default is ``nperseg // 2``.
    scaling : {'density', 'spectrum'}, optional
        Selects between cross power spectral density ('density') and
        cross spectrum ('spectrum'). Default is 'density'.

    Returns
    -------
    freqs : Tensor
        Frequency bins of shape ``(nperseg//2 + 1,)``.
    Pxy : Tensor
        Complex cross spectral density of shape ``(..., nperseg//2 + 1)``.

    Examples
    --------
    Compute CSD of two correlated signals:

    >>> torch.manual_seed(0)
    >>> t = torch.arange(10000, dtype=torch.float64) / 1000.0
    >>> x = torch.sin(2 * torch.pi * 50 * t)
    >>> y = torch.sin(2 * torch.pi * 50 * t + 0.5)
    >>> freqs, csd = cross_spectral_density(x, y, fs=1000.0)

    References
    ----------
    - Welch, P.D. "The use of fast Fourier transform for the estimation of
      power spectra." IEEE Trans. Audio Electroacoustics 15(2), 70-73 (1967).
    - Bendat, J.S. and Piersol, A.G. *Random Data: Analysis and Measurement
      Procedures*, 4th ed. Wiley, 2010.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if window is None:
        window = hann_window(nperseg, dtype=x.dtype, device=x.device)

    if scaling == "density":
        scaling_int = 0
    else:
        scaling_int = 1

    return torch.ops.torchscience.cross_spectral_density(
        x,
        y,
        window,
        nperseg,
        noverlap,
        fs,
        scaling_int,
    )
