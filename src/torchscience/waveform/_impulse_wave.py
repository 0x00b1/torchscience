from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def impulse_wave(
    n: int,
    *,
    position: Union[int, Tensor] = 0,
    amplitude: Union[float, Tensor] = 1.0,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Unit impulse (Dirac delta approximation) waveform generator.

    y[k] = amplitude if k == position else 0

    Parameters
    ----------
    n : int
        Number of samples.
    position : int or Tensor
        Sample index where impulse occurs. Tensor enables batched generation.
    amplitude : float or Tensor
        Impulse amplitude. Tensor enables batched generation.
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
        Impulse waveform of shape (n,) or (*batch_shape, n) if position/amplitude
        are tensors with batch dimensions.
    """
    # Convert scalars to tensors
    if not isinstance(position, Tensor):
        position = torch.tensor(position, dtype=torch.int64, device=device)
    else:
        position = position.to(dtype=torch.int64, device=device)

    if not isinstance(amplitude, Tensor):
        amplitude = torch.tensor(
            amplitude, dtype=dtype or torch.float32, device=device
        )

    if dtype is None:
        dtype = amplitude.dtype

    amplitude = amplitude.to(dtype=dtype, device=device)

    result = torch.ops.torchscience.impulse_wave(
        n,
        position=position,
        amplitude=amplitude,
        dtype=dtype,
        layout=layout,
        device=device,
    )

    if requires_grad and not result.requires_grad:
        result = result.requires_grad_(True)

    return result
