from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def bartlett_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Bartlett window function (symmetric).

    Computes a symmetric Bartlett (triangular) window of length n. The Bartlett
    window is equivalent to convolution of two rectangular windows, making it
    useful for smoothing applications.

    Mathematical Definition
    -----------------------
    The symmetric Bartlett window is defined as:

        w[k] = 1 - |k - (n-1)/2| / ((n-1)/2),  for k = 0, 1, ..., n-1

    which simplifies to a triangular shape with peak at the center.

    Properties
    ----------
    - Main lobe width: 8*pi/n
    - Side lobe level: -26.5 dB
    - Zero at endpoints
    - Linear slopes

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values.

    See Also
    --------
    periodic_bartlett_window : Periodic version for spectral analysis.
    """
    return torch.ops.torchscience.bartlett_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
