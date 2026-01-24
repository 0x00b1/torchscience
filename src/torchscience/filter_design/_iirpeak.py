"""Second-order IIR peak (resonator) filter design."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def iirpeak(
    peak_frequency: float,
    quality_factor: float,
    sampling_frequency: float = 2.0,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Design a second-order IIR peak (resonator) filter.

    A peak filter is a bandpass filter with a very narrow passband.
    It passes frequencies near the peak frequency while attenuating others.

    Parameters
    ----------
    peak_frequency : float
        Center frequency of the peak. If sampling_frequency is not
        specified (defaults to 2.0), this is normalized frequency in [0, 1)
        where 1 corresponds to the Nyquist frequency.
    quality_factor : float
        Quality factor Q that characterizes the peak filter. Higher Q means
        a narrower peak bandwidth. Q = peak_frequency / bandwidth.
        Common values: 10-50 for narrow peaks.
    sampling_frequency : float, optional
        The sampling frequency of the digital system. Default is 2.0
        (normalized frequency). If specified, peak_frequency should be
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
    The peak filter is the complement of the notch filter. Together,
    they satisfy: H_notch(z) + H_peak(z) = 1 for all z.

    The 3 dB bandwidth of the peak is approximately:

        bandwidth = peak_frequency / quality_factor

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import iirpeak
    >>> # Isolate 60 Hz from 1000 Hz sampled signal
    >>> b, a = iirpeak(60.0, quality_factor=30.0, sampling_frequency=1000.0)
    >>> b.shape, a.shape
    (torch.Size([3]), torch.Size([3]))

    >>> # Using normalized frequency (peak at 0.1 * Nyquist)
    >>> b, a = iirpeak(0.1, quality_factor=20.0)
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    # Validate inputs
    nyquist = sampling_frequency / 2.0
    if peak_frequency <= 0 or peak_frequency >= nyquist:
        raise ValueError(
            f"peak_frequency must be between 0 and Nyquist ({nyquist}), "
            f"got {peak_frequency}"
        )
    if quality_factor <= 0:
        raise ValueError(
            f"quality_factor must be positive, got {quality_factor}"
        )

    # Compute normalized angular frequency
    w0 = 2.0 * math.pi * peak_frequency / sampling_frequency

    # Compute bandwidth from quality factor
    bw = w0 / quality_factor

    # Compute filter coefficients using the bilinear transform approach
    # This follows scipy's implementation
    gb = 1.0 / math.sqrt(2.0)  # Gain at bandwidth edges (-3dB point)
    beta = (math.sqrt(1.0 - gb**2) / gb) * math.tan(bw / 2.0)

    gain = 1.0 / (1.0 + beta)

    # The peak filter is the complement of the notch filter:
    # H_peak(z) = 1 - H_notch(z)
    # So b_peak = [1,0,0] - b_notch (after normalization)

    # Numerator coefficients (peak is complement of notch)
    b0 = 1.0 - gain
    b1 = 0.0
    b2 = -(1.0 - gain)

    # Denominator coefficients (same as notch)
    a0 = 1.0
    a1 = -2.0 * gain * math.cos(w0)
    a2 = 2.0 * gain - 1.0

    b = torch.tensor([b0, b1, b2], dtype=dtype, device=device)
    a = torch.tensor([a0, a1, a2], dtype=dtype, device=device)

    return b, a
