"""Chebyshev Type I analog lowpass filter prototype."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def chebyshev_type_1_prototype(
    order: int,
    passband_ripple_db: float,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Design an analog Chebyshev Type I lowpass filter prototype.

    Returns the zeros, poles, and gain of a normalized analog Chebyshev Type I
    lowpass filter with the specified passband ripple. The filter has equiripple
    passband and monotonic stopband.

    Parameters
    ----------
    order : int
        The order of the filter. Must be positive.
    passband_ripple_db : float
        Maximum ripple in the passband in decibels. Must be positive.
        Common values: 0.5 dB, 1 dB, 3 dB.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    zeros : Tensor
        Zeros of the filter (empty for Chebyshev Type I).
    poles : Tensor
        Poles of the filter, complex tensor of shape (order,).
    gain : Tensor
        System gain, scalar tensor.

    Notes
    -----
    The Chebyshev Type I filter has the property that the magnitude response
    oscillates between 1 and 1/(1+eps^2) in the passband, where:

    .. math::
        \\epsilon = \\sqrt{10^{R_p/10} - 1}

    and R_p is the passband ripple in dB.

    The poles are located at:

    .. math::
        p_k = \\sinh(a) \\sin(\\theta_k) + j \\cosh(a) \\cos(\\theta_k)

    where:
    - a = (1/n) * arcsinh(1/eps)
    - theta_k = pi * (2k + 1) / (2n) for k = 0, 1, ..., n-1

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import chebyshev_type_1_prototype
    >>> zeros, poles, gain = chebyshev_type_1_prototype(4, passband_ripple_db=1.0)
    >>> poles.shape
    torch.Size([4])
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")
    if passband_ripple_db <= 0:
        raise ValueError(
            f"Passband ripple must be positive, got {passband_ripple_db}"
        )

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

    # Compute epsilon from ripple: eps = sqrt(10^(Rp/10) - 1)
    eps = math.sqrt(10 ** (passband_ripple_db / 10) - 1)

    # Compute the parameter a = (1/n) * arcsinh(1/eps)
    a = math.asinh(1.0 / eps) / order

    # Compute poles
    # theta_k = pi * (2k + 1) / (2n) for k = 0, 1, ..., n-1
    k = torch.arange(order, dtype=torch.float64, device=device)
    theta = math.pi * (2 * k + 1) / (2 * order)

    # Poles: p_k = -sinh(a)*sin(theta_k) + j*cosh(a)*cos(theta_k)
    # Note: negative real part for stability (left half-plane)
    real_part = -math.sinh(a) * torch.sin(theta)
    imag_part = math.cosh(a) * torch.cos(theta)

    poles = torch.complex(real_part, imag_part)

    # No zeros for Chebyshev Type I (all-pole filter)
    zeros = torch.zeros(0, dtype=complex_dtype, device=device)

    # Compute gain
    # For odd order: gain = real(prod(-poles))
    # For even order: gain = real(prod(-poles)) / sqrt(1 + eps^2)
    gain_complex = torch.prod(-poles)
    gain = gain_complex.real

    if order % 2 == 0:
        # Even order: normalize to 1/sqrt(1+eps^2) at DC
        gain = gain / math.sqrt(1 + eps**2)

    # Convert to requested dtype (poles stay complex)
    return zeros, poles.to(complex_dtype), gain.to(dtype)
