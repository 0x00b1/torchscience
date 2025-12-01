from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def hann_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Hann window function (symmetric).

    Computes a symmetric Hann window of length n. The Hann window is a raised
    cosine window that tapers smoothly to zero at the endpoints, reducing
    spectral leakage in Fourier analysis.

    Mathematical Definition
    -----------------------
    The symmetric Hann window is defined as:

        w[k] = 0.5 * (1 - cos(2 * pi * k / (n - 1))),  for k = 0, 1, ..., n-1

    Properties
    ----------
    - Main lobe width: 8*pi/n
    - Side lobe level: -31.5 dB
    - The window is exactly zero at the endpoints
    - Smooth first derivative at endpoints

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
    periodic_hann_window : Periodic version for spectral analysis with FFT.
    """
    return torch.ops.torchscience.hann_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
