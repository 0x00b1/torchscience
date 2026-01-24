from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def pulse_wave(
    n: Optional[int] = None,
    t: Optional[Tensor] = None,
    *,
    frequency: Union[float, Tensor] = 1.0,
    sample_rate: float = 1.0,
    amplitude: Union[float, Tensor] = 1.0,
    phase: Union[float, Tensor] = 0.0,
    duty_cycle: Union[float, Tensor] = 0.5,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Pulse waveform with differentiable duty cycle.

    A pulse wave is a rectangular waveform where the duty cycle controls
    the fraction of each period spent at the positive amplitude.

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
    duty_cycle : float or Tensor
        Duty cycle in [0, 1]. Fraction of period at +amplitude.

    Returns
    -------
    Tensor
        Pulse waveform.
    """
    if n is not None and t is not None:
        raise ValueError("n and t are mutually exclusive")
    if n is None and t is None:
        raise ValueError("Either n or t must be provided")

    # Convert scalars to tensors
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
    if not isinstance(duty_cycle, Tensor):
        duty_cycle = torch.tensor(
            duty_cycle, dtype=dtype or torch.float32, device=device
        )

    if dtype is None:
        dtype = torch.result_type(frequency, amplitude)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), phase)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), duty_cycle)
        if t is not None:
            dtype = torch.result_type(torch.empty(0, dtype=dtype), t)

    frequency = frequency.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)
    phase = phase.to(dtype=dtype, device=device)
    duty_cycle = duty_cycle.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.pulse_wave(
        n,
        t,
        frequency=frequency,
        sample_rate=sample_rate,
        amplitude=amplitude,
        phase=phase,
        duty_cycle=duty_cycle,
        dtype=dtype,
        layout=layout,
        device=device,
    )

    if requires_grad and not result.requires_grad:
        result = result.requires_grad_(True)

    return result
