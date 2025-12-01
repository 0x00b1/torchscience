from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def blackman_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Blackman window function (symmetric).

    Computes a symmetric Blackman window of length n. The Blackman window
    provides excellent side lobe suppression at the cost of a wider main lobe,
    making it suitable for applications requiring minimal spectral leakage.

    Mathematical Definition
    -----------------------
    The symmetric Blackman window is defined as:

        w[k] = 0.42 - 0.5 * cos(2 * pi * k / (n-1)) + 0.08 * cos(4 * pi * k / (n-1))

    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - Main lobe width: 12*pi/n
    - Side lobe level: -58 dB
    - Zero at endpoints

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
    periodic_blackman_window : Periodic version for spectral analysis.
    """
    return torch.ops.torchscience.blackman_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
