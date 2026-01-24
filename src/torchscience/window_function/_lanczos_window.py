from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def lanczos_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Lanczos window function (symmetric).

    Computes a symmetric Lanczos window of length n. The Lanczos window, also
    known as the sinc window, is defined using the normalized sinc function.
    It provides smooth tapering and is commonly used in signal processing and
    image resampling applications.

    Mathematical Definition
    -----------------------
    The symmetric Lanczos window is defined as:

        w[k] = sinc(2k / (n - 1) - 1),  for k = 0, 1, ..., n-1

    where sinc(x) = sin(pi * x) / (pi * x) with sinc(0) = 1.

    Properties
    ----------
    - Smooth sinc-based shape
    - Zero at endpoints (for n > 2)
    - Maximum value of 1.0 at the center
    - Good balance between main lobe width and side lobe suppression

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
    periodic_lanczos_window : Periodic version for spectral analysis.

    Notes
    -----
    The Lanczos window is widely used in image processing for high-quality
    resampling (Lanczos resampling). It offers a good trade-off between
    sharpness and ringing artifacts.
    """
    return torch.ops.torchscience.lanczos_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
