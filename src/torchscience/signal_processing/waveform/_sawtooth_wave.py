from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def sawtooth_wave(
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
    Sawtooth waveform generator.

    y[k] = amplitude * (2 * frac(frequency * t[k] + phase/(2*pi)) - 1)

    Parameters
    ----------
    n : int, optional
        Number of samples. Mutually exclusive with t.
    t : Tensor, optional
        Explicit time tensor. Mutually exclusive with n.
    frequency : float or Tensor
        Frequency in Hz.
    sample_rate : float
        Samples per second. Ignored when t is provided.
    amplitude : float or Tensor
        Peak amplitude.
    phase : float or Tensor
        Initial phase in radians.

    Returns
    -------
    Tensor
        Sawtooth waveform.
    """
    if n is not None and t is not None:
        raise ValueError("n and t are mutually exclusive")
    if n is None and t is None:
        raise ValueError("Either n or t must be provided")

    if not isinstance(frequency, Tensor):
        frequency = torch.tensor(
            frequency, dtype=dtype or torch.float32, device=device
        )
    if not isinstance(amplitude, Tensor):
        amplitude = torch.tensor(
            amplitude, dtype=dtype or torch.float32, device=device
        )
    if not isinstance(phase, Tensor):
        phase = torch.tensor(
            phase, dtype=dtype or torch.float32, device=device
        )

    if dtype is None:
        dtype = torch.result_type(frequency, amplitude)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), phase)
        if t is not None:
            dtype = torch.result_type(torch.empty(0, dtype=dtype), t)

    frequency = frequency.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)
    phase = phase.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.sawtooth_wave(
        n,
        t,
        frequency=frequency,
        sample_rate=sample_rate,
        amplitude=amplitude,
        phase=phase,
        dtype=dtype,
        layout=layout,
        device=device,
    )

    if requires_grad and not result.requires_grad:
        result = result.requires_grad_(True)

    return result
