from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def gaussian_pulse_wave(
    n: int,
    *,
    center: Union[float, Tensor] = 0.0,
    std: Union[float, Tensor] = 1.0,
    amplitude: Union[float, Tensor] = 1.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Gaussian pulse waveform generator.

    Generates a Gaussian pulse (transient waveform) centered at a specified
    position with given standard deviation and amplitude.

    y[k] = amplitude * exp(-0.5 * ((k - center) / std)^2)

    Parameters
    ----------
    n : int
        Number of samples.
    center : float or Tensor
        Center position of the Gaussian pulse (can be non-integer).
        Tensor enables batched generation.
    std : float or Tensor
        Standard deviation of the Gaussian pulse.
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
        Gaussian pulse waveform of shape (n,) or (*batch_shape, n) if
        center/std/amplitude are tensors with batch dimensions.
    """
    # Convert scalars to tensors
    if not isinstance(center, Tensor):
        center = torch.tensor(
            center, dtype=dtype or torch.float32, device=device
        )

    if not isinstance(std, Tensor):
        std = torch.tensor(std, dtype=dtype or torch.float32, device=device)

    if not isinstance(amplitude, Tensor):
        amplitude = torch.tensor(
            amplitude, dtype=dtype or torch.float32, device=device
        )

    if dtype is None:
        dtype = amplitude.dtype

    center = center.to(dtype=dtype, device=device)
    std = std.to(dtype=dtype, device=device)
    amplitude = amplitude.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.gaussian_pulse_wave(
        n,
        center=center,
        std=std,
        amplitude=amplitude,
        dtype=dtype,
        layout=layout,
        device=device,
    )

    if requires_grad and not result.requires_grad:
        result = result.requires_grad_(True)

    return result
