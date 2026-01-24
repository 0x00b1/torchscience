"""Butterworth analog lowpass filter prototype."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def butterworth_prototype(
    order: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Butterworth analog lowpass filter prototype.

    Returns the zeros, poles, and gain of an nth-order normalized analog
    lowpass Butterworth filter prototype with cutoff frequency of 1 rad/s.

    Parameters
    ----------
    order : int
        Filter order. Must be positive.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    zeros : Tensor
        Zeros of the filter (empty for Butterworth).
    poles : Tensor
        Poles of the filter, shape (order,), complex dtype.
    gain : Tensor
        System gain (scalar, equals 1.0 for normalized prototype).

    Notes
    -----
    The Butterworth filter has poles located on the unit circle in the
    left half of the s-plane, equally spaced in angle:

    .. math::
        s_k = e^{j \\pi (2k + n + 1) / (2n)} \\quad \\text{for } k = 0, \\ldots, n-1

    The filter has no finite zeros (all zeros at infinity).

    Examples
    --------
    >>> zeros, poles, gain = butterworth_prototype(4)
    >>> poles.shape
    torch.Size([4])
    >>> gain
    tensor(1.)

    References
    ----------
    .. [1] S. Butterworth, "On the Theory of Filter Amplifiers,"
           Wireless Engineer, vol. 7, pp. 536-541, 1930.
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")

    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")

    # Determine complex dtype
    if dtype == torch.float32:
        complex_dtype = torch.complex64
    elif dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Zeros: empty for Butterworth (all zeros at infinity)
    zeros = torch.empty(0, dtype=complex_dtype, device=device)

    # Poles: on unit circle in left half-plane
    # s_k = exp(j * pi * (2k + n + 1) / (2n)) for k = 0, ..., n-1
    k_indices = torch.arange(0, order, dtype=dtype, device=device)
    angles = math.pi * (2 * k_indices + order + 1) / (2 * order)

    # Compute poles as complex exponentials
    poles = torch.complex(torch.cos(angles), torch.sin(angles))

    # Gain: 1.0 for normalized prototype
    gain = torch.tensor(1.0, dtype=dtype, device=device)

    return zeros, poles, gain
