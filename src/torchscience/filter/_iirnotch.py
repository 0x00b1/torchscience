"""Second-order IIR notch filter design."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def iirnotch(
    notch_frequency: float,
    quality_factor: float,
    sampling_frequency: float = 2.0,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Design a second-order IIR notch (band-reject) filter.

    A notch filter is a band-reject filter with a very narrow reject band.
    It attenuates frequencies near the notch frequency while passing all others.

    Parameters
    ----------
    notch_frequency : float
        Frequency to remove from the signal. If sampling_frequency is not
        specified (defaults to 2.0), this is normalized frequency in [0, 1)
        where 1 corresponds to the Nyquist frequency.
    quality_factor : float
        Quality factor Q that characterizes the notch filter. Higher Q means
        a narrower notch bandwidth. Q = notch_frequency / bandwidth.
        Common values: 10-50 for narrow notches.
    sampling_frequency : float, optional
        The sampling frequency of the digital system. Default is 2.0
        (normalized frequency). If specified, notch_frequency should be
        in the same units (e.g., Hz).
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    b : Tensor
        Numerator coefficients of the filter, shape (3,).
    a : Tensor
        Denominator coefficients of the filter, shape (3,).

    Notes
    -----
    The notch filter is designed using the formula:

        H(z) = (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)

    where w0 = 2*pi*notch_frequency/sampling_frequency is the notch frequency
    in radians, and r is determined by the quality factor.

    The 3 dB bandwidth of the notch is approximately:

        bandwidth = notch_frequency / quality_factor

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import iirnotch
    >>> # Remove 60 Hz hum from 1000 Hz sampled signal
    >>> b, a = iirnotch(60.0, quality_factor=30.0, sampling_frequency=1000.0)
    >>> b.shape, a.shape
    (torch.Size([3]), torch.Size([3]))

    >>> # Using normalized frequency (notch at 0.1 * Nyquist)
    >>> b, a = iirnotch(0.1, quality_factor=20.0)
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    # Validate inputs
    nyquist = sampling_frequency / 2.0
    if notch_frequency <= 0 or notch_frequency >= nyquist:
        raise ValueError(
            f"notch_frequency must be between 0 and Nyquist ({nyquist}), "
            f"got {notch_frequency}"
        )
    if quality_factor <= 0:
        raise ValueError(
            f"quality_factor must be positive, got {quality_factor}"
        )

    # Compute normalized angular frequency
    w0 = 2.0 * math.pi * notch_frequency / sampling_frequency

    # Compute bandwidth from quality factor
    # bandwidth = notch_frequency / quality_factor
    # Using the bilinear transform formulation
    bw = w0 / quality_factor

    # Compute filter coefficients using the bilinear transform approach
    # This follows scipy's implementation
    gb = 1.0 / math.sqrt(2.0)  # Gain at bandwidth edges (-3dB point)
    beta = (math.sqrt(1.0 - gb**2) / gb) * math.tan(bw / 2.0)

    gain = 1.0 / (1.0 + beta)

    # Numerator coefficients
    b0 = gain
    b1 = -2.0 * gain * math.cos(w0)
    b2 = gain

    # Denominator coefficients
    a0 = 1.0
    a1 = -2.0 * gain * math.cos(w0)
    a2 = 2.0 * gain - 1.0

    b = torch.tensor([b0, b1, b2], dtype=dtype, device=device)
    a = torch.tensor([a0, a1, a2], dtype=dtype, device=device)

    return b, a
