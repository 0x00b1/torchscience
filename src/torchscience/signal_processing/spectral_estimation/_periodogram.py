from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodogram(
    x: Tensor,
    *,
    fs: float = 1.0,
    window: Optional[Tensor] = None,
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[Tensor, Tensor]:
    r"""
    Power spectral density estimate using a periodogram.

    Estimates the power spectral density of a real-valued signal using the
    squared magnitude of its discrete Fourier transform.

    Mathematical Definition
    -----------------------
    For a signal :math:`x[n]` of length :math:`N` with window :math:`w[n]`:

    .. math::

        P(f_k) = \frac{1}{f_s \cdot S} \left| \sum_{n=0}^{N-1} x[n] \, w[n] \, e^{-j 2\pi k n / N} \right|^2

    where :math:`S = \sum_{n=0}^{N-1} w[n]^2` for density scaling and
    :math:`S = \left(\sum_{n=0}^{N-1} w[n]\right)^2` for spectrum scaling.

    Parameters
    ----------
    x : Tensor
        Input signal tensor of shape ``(..., N)``. The last dimension is
        treated as the signal axis. Batch dimensions are fully supported.
    fs : float, optional
        Sampling frequency of the signal. Default is 1.0.
    window : Tensor, optional
        Window tensor of shape ``(N,)``. If ``None``, a rectangular window
        (all ones) is used.
    scaling : {'density', 'spectrum'}, optional
        Selects between computing the power spectral density ('density')
        where ``Pxx`` has units of V**2/Hz, and computing the squared
        magnitude spectrum ('spectrum') where ``Pxx`` has units of V**2.
        Default is 'density'.

    Returns
    -------
    freqs : Tensor
        Frequency bins of shape ``(N//2 + 1,)``.
    Pxx : Tensor
        Power spectral density or spectrum of shape ``(..., N//2 + 1)``.

    Examples
    --------
    Estimate the PSD of a sine wave:

    >>> t = torch.linspace(0, 1, 1000)
    >>> x = torch.sin(2 * torch.pi * 50 * t)  # 50 Hz sine
    >>> freqs, psd = periodogram(x, fs=1000.0)
    >>> freqs[psd.argmax()]
    tensor(50.)

    References
    ----------
    - Schuster, A. "On the investigation of hidden periodicities."
      Terrestrial Magnetism 3, 13-41 (1898).
    - Oppenheim, A.V. and Schafer, R.W. *Discrete-Time Signal Processing*,
      3rd ed. Pearson, 2010.
    """
    if window is None:
        window = torch.ones(x.size(-1), dtype=x.dtype, device=x.device)

    scaling_int = 0 if scaling == "density" else 1

    return torch.ops.torchscience.periodogram(x, window, fs, scaling_int)
