from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def linear_chirp_wave(
    n: Optional[int] = None,
    t: Optional[Tensor] = None,
    *,
    f0: Union[float, Tensor] = 1.0,
    f1: Union[float, Tensor] = 10.0,
    t1: float = 1.0,
    sample_rate: float = 1.0,
    amplitude: Union[float, Tensor] = 1.0,
    phase: Union[float, Tensor] = 0.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Linear chirp (frequency sweep) waveform generator.

    Generates a chirp signal where the instantaneous frequency varies linearly
    from f0 to f1 over the time interval [0, t1]:

        f(t) = f0 + (f1 - f0) * t / t1

    The output signal is:

        y(t) = amplitude * cos(2*pi*(f0*t + (f1-f0)*t^2/(2*t1)) + phase)

    Parameters
    ----------
    n : int, optional
        Number of samples. Mutually exclusive with t.
    t : Tensor, optional
        Explicit time tensor. Mutually exclusive with n.
    f0 : float or Tensor
        Starting frequency in Hz. Tensor enables batched generation.
    f1 : float or Tensor
        Ending frequency in Hz (at time t1). Tensor enables batched generation.
    t1 : float
        Time at which frequency reaches f1. Default is 1.0.
    sample_rate : float
        Samples per second. Ignored when t is provided. Default is 1.0.
    amplitude : float or Tensor
        Peak amplitude. Tensor enables batched generation. Default is 1.0.
    phase : float or Tensor
        Initial phase in radians. Tensor enables batched generation. Default is 0.0.
    dtype : torch.dtype, optional
        Output dtype.
    layout : torch.layout, optional
        Output layout.
    device : torch.device, optional
        Output device.
    requires_grad : bool
        If True, enables gradient computation. Default is False.

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
    Generate a basic linear chirp from 1 Hz to 10 Hz:

    >>> linear_chirp_wave(n=1000, f0=1.0, f1=10.0, sample_rate=1000.0)

    Generate batched chirps with different starting frequencies:

    >>> f0 = torch.tensor([1.0, 5.0, 10.0])
    >>> linear_chirp_wave(n=1000, f0=f0, f1=100.0, sample_rate=1000.0)  # shape (3, 1000)

    Use explicit time tensor:

    >>> t = torch.linspace(0, 1, 1000)
    >>> linear_chirp_wave(t=t, f0=1.0, f1=50.0)

    Notes
    -----
    The linear chirp is commonly used in radar and sonar systems, as well as
    in audio synthesis. The scipy.signal.chirp function provides a similar
    capability with method='linear'.

    The instantaneous phase is computed as:

        phi(t) = 2*pi * (f0*t + (f1-f0)*t^2/(2*t1)) + phase

    which corresponds to integrating the instantaneous frequency.
    """
    # Validate mutual exclusivity
    if n is not None and t is not None:
        raise ValueError("n and t are mutually exclusive - provide only one")
    if n is None and t is None:
        raise ValueError("Either n or t must be provided")

    # Convert scalars to 0-D tensors
    if not isinstance(f0, Tensor):
        f0 = torch.tensor(f0, dtype=dtype or torch.float32, device=device)
    if not isinstance(f1, Tensor):
        f1 = torch.tensor(f1, dtype=dtype or torch.float32, device=device)
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
        dtype = torch.result_type(f0, f1)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), amplitude)
        dtype = torch.result_type(torch.empty(0, dtype=dtype), phase)
        if t is not None:
            dtype = torch.result_type(torch.empty(0, dtype=dtype), t)

    f0 = f0.to(dtype=dtype, device=device)
    f1 = f1.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)
    phase = phase.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.linear_chirp_wave(
        n,
        t,
        f0=f0,
        f1=f1,
        t1=t1,
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
