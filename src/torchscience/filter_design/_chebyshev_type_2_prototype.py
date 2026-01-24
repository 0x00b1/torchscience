"""Chebyshev Type II analog lowpass filter prototype."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def chebyshev_type_2_prototype(
    order: int,
    stopband_attenuation_db: float,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Design an analog Chebyshev Type II lowpass filter prototype.

    Returns the zeros, poles, and gain of a normalized analog Chebyshev Type II
    lowpass filter with the specified stopband attenuation. The filter has
    monotonic passband and equiripple stopband.

    Parameters
    ----------
    order : int
        The order of the filter. Must be positive.
    stopband_attenuation_db : float
        Minimum attenuation in the stopband in decibels. Must be positive.
        Common values: 20 dB, 40 dB, 60 dB.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.get_default_dtype().
    device : torch.device, optional
        Output device. Defaults to CPU.

    Returns
    -------
    zeros : Tensor
        Zeros of the filter (on imaginary axis), complex tensor.
    poles : Tensor
        Poles of the filter, complex tensor of shape (order,).
    gain : Tensor
        System gain, scalar tensor.

    Notes
    -----
    The Chebyshev Type II filter (also called inverse Chebyshev) has:
    - Monotonically decreasing passband (like Butterworth)
    - Equiripple stopband with attenuation oscillating at the specified level

    The zeros are located at:

    .. math::
        z_k = j / \\cos(\\theta_k)

    where theta_k = pi * (2k + 1) / (2n) for k = 0, 1, ..., n-1.

    For odd order, there is no zero at infinity (one fewer zero than poles).

    The poles are the reciprocals of the Chebyshev Type I poles, inverted
    and scaled to maintain the passband edge at w=1.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import chebyshev_type_2_prototype
    >>> zeros, poles, gain = chebyshev_type_2_prototype(4, stopband_attenuation_db=40.0)
    >>> zeros.shape, poles.shape
    (torch.Size([4]), torch.Size([4]))
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")
    if stopband_attenuation_db <= 0:
        raise ValueError(
            f"Stopband attenuation must be positive, got {stopband_attenuation_db}"
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

    # Compute epsilon from stopband attenuation
    # Rs = 20*log10(1/delta_s), so delta_s = 10^(-Rs/20)
    # eps = 1/sqrt(10^(Rs/10) - 1)
    eps = 1.0 / math.sqrt(10 ** (stopband_attenuation_db / 10) - 1)

    # Compute the parameter a = (1/n) * arcsinh(1/eps)
    a = math.asinh(1.0 / eps) / order

    # Compute Chebyshev Type I poles (before inversion)
    k = torch.arange(order, dtype=torch.float64, device=device)
    theta = math.pi * (2 * k + 1) / (2 * order)

    # Type I poles
    sinh_a = math.sinh(a)
    cosh_a = math.cosh(a)
    p1_real = -sinh_a * torch.sin(theta)
    p1_imag = cosh_a * torch.cos(theta)
    p1 = torch.complex(p1_real, p1_imag)

    # Type II poles are 1/p1 (reciprocal)
    poles = 1.0 / p1

    # Zeros are on imaginary axis at 1/(j*cos(theta_k))
    # = -j/cos(theta_k) = j * (-1/cos(theta_k))
    # For n even: n zeros
    # For n odd: n-1 zeros (no zero at infinity, pole at real axis)
    cos_theta = torch.cos(theta)

    # For odd order, the middle theta gives cos(theta) = 0, causing division by zero
    # This corresponds to a zero at infinity (no finite zero)
    if order % 2 == 1:
        # Remove the middle element (where cos(theta) = 0)
        mid = order // 2
        cos_theta_valid = torch.cat([cos_theta[:mid], cos_theta[mid + 1 :]])
        zeros_imag = 1.0 / cos_theta_valid
    else:
        zeros_imag = 1.0 / cos_theta

    zeros = torch.complex(torch.zeros_like(zeros_imag), zeros_imag)

    # Compute gain to normalize DC response to 1
    # gain = real(prod(-poles) / prod(-zeros)) for even order
    # For odd order, there's one fewer zero
    num = torch.prod(-poles)
    if zeros.numel() > 0:
        den = torch.prod(-zeros)
        gain = (num / den).real
    else:
        gain = num.real

    # Convert to requested dtype (zeros and poles stay complex)
    return zeros.to(complex_dtype), poles.to(complex_dtype), gain.to(dtype)
