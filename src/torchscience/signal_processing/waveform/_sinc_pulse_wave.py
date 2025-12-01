from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def sinc_pulse_wave(
    n: int,
    *,
    center: Union[float, Tensor] = 0.0,
    bandwidth: Union[float, Tensor] = 1.0,
    amplitude: Union[float, Tensor] = 1.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Sinc pulse waveform generator.

    Generates a normalized sinc pulse (transient waveform) centered at a
    specified position with given bandwidth and amplitude.

    y[k] = amplitude * sinc(bandwidth * (k - center))

    where sinc(x) = sin(pi*x) / (pi*x) is the normalized sinc function,
    which equals 1 at x=0.

    Parameters
    ----------
    n : int
        Number of samples.
    center : float or Tensor
        Center position of the sinc pulse (can be non-integer).
        Tensor enables batched generation.
    bandwidth : float or Tensor
        Bandwidth of the sinc pulse. Higher values produce narrower pulses.
        Tensor enables batched generation.
    amplitude : float or Tensor
        Peak amplitude of the pulse. Tensor enables batched generation.
    dtype : torch.dtype, optional
        The desired data type of returned tensor.
    layout : torch.layout, optional
        The desired layout of returned tensor.
    device : torch.device, optional
        The desired device of returned tensor.
    requires_grad : bool
        If autograd should record operations on the returned tensor.

    Returns
    -------
    Tensor
        Sinc pulse waveform of shape (n,) or (*batch_shape, n) if
        center/bandwidth/amplitude are tensors with batch dimensions.
    """
    # Convert scalars to tensors
    if not isinstance(center, Tensor):
        center = torch.tensor(
            center, dtype=dtype or torch.float32, device=device
        )

    if not isinstance(bandwidth, Tensor):
        bandwidth = torch.tensor(
            bandwidth, dtype=dtype or torch.float32, device=device
        )

    if not isinstance(amplitude, Tensor):
        amplitude = torch.tensor(
            amplitude, dtype=dtype or torch.float32, device=device
        )

    if dtype is None:
        dtype = amplitude.dtype

    center = center.to(dtype=dtype, device=device)
    bandwidth = bandwidth.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.sinc_pulse_wave(
        n,
        center=center,
        bandwidth=bandwidth,
        amplitude=amplitude,
        dtype=dtype,
        layout=layout,
        device=device,
    )

    if requires_grad and not result.requires_grad:
        result = result.requires_grad_(True)

    return result
