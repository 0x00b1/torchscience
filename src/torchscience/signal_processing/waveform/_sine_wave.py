from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def sine_wave(
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
    Sinusoidal waveform generator.

    y[k] = amplitude * sin(2π * frequency * t[k] + phase)

    Parameters
    ----------
    n : int, optional
        Number of samples. Mutually exclusive with t.
    t : Tensor, optional
        Explicit time tensor. Mutually exclusive with n.
    frequency : float or Tensor
        Frequency in Hz. Tensor enables batched generation.
    sample_rate : float
        Samples per second. Ignored when t is provided.
    amplitude : float or Tensor
        Peak amplitude. Tensor enables batched generation.
    phase : float or Tensor
        Initial phase in radians. Tensor enables batched generation.
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
        Shape (*broadcast_shape, n) or (*broadcast_shape, *t.shape)

    Raises
    ------
    ValueError
        If both n and t are provided, or neither is provided.

    Examples
    --------
    Generate a basic sine wave:
    >>> sine_wave(n=100, frequency=1.0, sample_rate=100.0)

    Generate batched sine waves:
    >>> freqs = torch.tensor([220.0, 440.0, 880.0])
    >>> sine_wave(n=1000, frequency=freqs, sample_rate=44100.0)  # shape (3, 1000)

    Use explicit time tensor:
    >>> t = torch.linspace(0, 1, 1000)
    >>> sine_wave(t=t, frequency=440.0)
    """
    # Validate mutual exclusivity
    if n is not None and t is not None:
        raise ValueError("n and t are mutually exclusive - provide only one")
    if n is None and t is None:
        raise ValueError("Either n or t must be provided")

    # Convert scalars to 0-D tensors
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

    # Ensure consistent dtype across parameters
    if dtype is None:
        dtype = torch.result_type(frequency, amplitude)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), phase)
        if t is not None:
            dtype = torch.result_type(torch.empty(0, dtype=dtype), t)

    frequency = frequency.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)
    phase = phase.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.sine_wave(
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
