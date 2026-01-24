import math
from typing import Optional, Union

import torch
from torch import Tensor

from ._sine_wave import sine_wave


def cosine_wave(
    n: Optional[int] = None,
    t: Optional[Tensor] = None,
    *,
    frequency: Union[float, Tensor] = 1.0,
    sample_rate: float = 1.0,
    amplitude: Union[float, Tensor] = 1.0,
    phase: Union[float, Tensor] = 0.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Cosine waveform generator.

    Equivalent to sine_wave with phase + pi/2.

    y[k] = amplitude * cos(2pi * frequency * t[k] + phase)

    Parameters
    ----------
    n : int, optional
        Number of samples. Mutually exclusive with t.
    t : Tensor, optional
        Explicit time tensor. Mutually exclusive with n.
    frequency : float or Tensor
        Frequency in Hz.
    sample_rate : float
        Samples per second.
    amplitude : float or Tensor
        Peak amplitude.
    phase : float or Tensor
        Initial phase in radians.
    dtype : torch.dtype, optional
        Output dtype.
    layout : torch.layout, optional
        Output layout.
    device : torch.device, optional
        Output device.
    requires_grad : bool
        If True, enables gradient computation.

    Returns
    -------
    Tensor
        Cosine waveform.

    Examples
    --------
    Generate a basic cosine wave:
    >>> cosine_wave(n=100, frequency=1.0, sample_rate=100.0)

    Generate batched cosine waves:
    >>> freqs = torch.tensor([220.0, 440.0, 880.0])
    >>> cosine_wave(n=1000, frequency=freqs, sample_rate=44100.0)  # shape (3, 1000)
    """
    adjusted_phase = phase + math.pi / 2

    return sine_wave(
        n=n,
        t=t,
        frequency=frequency,
        sample_rate=sample_rate,
        amplitude=amplitude,
        phase=adjusted_phase,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
