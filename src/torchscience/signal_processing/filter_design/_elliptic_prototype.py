"""Elliptic (Cauer) analog lowpass filter prototype."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._elliptic_functions import elliptic_prototype as _elliptic_prototype_impl


def elliptic_prototype(
    order: int,
    passband_ripple_db: float,
    stopband_attenuation_db: float,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Design an analog elliptic (Cauer) lowpass filter prototype.

    Returns the zeros, poles, and gain of a normalized analog elliptic
    lowpass filter with the specified passband ripple and stopband attenuation.
    Elliptic filters provide the steepest rolloff for a given filter order
    at the cost of ripple in both passband and stopband.

    Parameters
    ----------
    order : int
        The order of the filter. Must be positive.
    passband_ripple_db : float
        Maximum ripple in the passband in decibels. Must be positive.
        Common values: 0.5 dB, 1 dB, 3 dB.
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
    The elliptic filter (also called Cauer or Zolotarev filter) has:
    - Equiripple passband with specified maximum ripple
    - Equiripple stopband with specified minimum attenuation
    - Sharpest possible transition band for given order and ripple specs

    This makes it the optimal choice when strict passband and stopband
    specifications must be met with the lowest possible filter order.

    The zeros are located on the imaginary axis, and poles are in the
    left half-plane. For even order filters, there are n zeros and n poles.
    For odd order filters, there are n-1 zeros and n poles (one real pole).

    Examples
    --------
    >>> import torch
    >>> from torchscience.signal_processing.filter_design import elliptic_prototype
    >>> zeros, poles, gain = elliptic_prototype(4, passband_ripple_db=1.0, stopband_attenuation_db=40.0)
    >>> zeros.shape, poles.shape
    (torch.Size([4]), torch.Size([4]))
    """
    if order < 1:
        raise ValueError(f"Filter order must be positive, got {order}")
    if passband_ripple_db <= 0:
        raise ValueError(
            f"Passband ripple must be positive, got {passband_ripple_db}"
        )
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

    # Get zeros, poles, gain from implementation
    zeros_list, poles_list, gain_val = _elliptic_prototype_impl(
        order, passband_ripple_db, stopband_attenuation_db
    )

    # Convert to tensors
    if zeros_list:
        zeros = torch.tensor(
            [[z.real, z.imag] for z in zeros_list],
            dtype=torch.float64,
            device=device,
        )
        zeros = torch.complex(zeros[:, 0], zeros[:, 1]).to(complex_dtype)
    else:
        zeros = torch.zeros(0, dtype=complex_dtype, device=device)

    poles = torch.tensor(
        [[p.real, p.imag] for p in poles_list],
        dtype=torch.float64,
        device=device,
    )
    poles = torch.complex(poles[:, 0], poles[:, 1]).to(complex_dtype)

    gain = torch.tensor(gain_val, dtype=dtype, device=device)

    return zeros, poles, gain
