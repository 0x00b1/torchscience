from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def triangular_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Triangular window function (symmetric).

    Computes a symmetric triangular window of length n. Unlike the Bartlett
    window which has zero endpoints, the triangular window has non-zero
    endpoints.

    Mathematical Definition
    -----------------------
    The symmetric triangular window is defined as:

    For odd n:
        w[k] = 2*(k+1)/(n+1)       for k < n/2
        w[k] = 2*(n-k)/(n+1)       for k >= n/2

    For even n:
        w[k] = (2*k+1)/n           for k < n/2
        w[k] = (2*(n-k)-1)/n       for k >= n/2

    Properties
    ----------
    - Main lobe width: 8*pi/n
    - Side lobe level: -26.5 dB
    - Non-zero endpoints (unlike Bartlett)
    - Linear slopes from edges to center

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
    periodic_triangular_window : Periodic version for spectral analysis with FFT.
    bartlett_window : Similar but with zero endpoints.
    """
    return torch.ops.torchscience.triangular_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
