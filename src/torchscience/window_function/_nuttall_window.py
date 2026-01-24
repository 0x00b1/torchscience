from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def nuttall_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Nuttall window function (symmetric).

    Computes a symmetric Nuttall window of length n. The Nuttall window is a
    4-term Blackman-Harris variant with minimum side lobes, optimized for
    applications requiring very low spectral leakage.

    Mathematical Definition
    -----------------------
    The symmetric Nuttall window is defined as:

        w[k] = a0 - a1*cos(2*pi*k/(n-1)) + a2*cos(4*pi*k/(n-1)) - a3*cos(6*pi*k/(n-1))

    for k = 0, 1, ..., n-1, where:
        a0 = 0.355768, a1 = 0.487396, a2 = 0.144232, a3 = 0.012604

    Properties
    ----------
    - Main lobe width: 16*pi/n
    - Side lobe level: -93.3 dB
    - Excellent spectral leakage suppression

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
    periodic_nuttall_window : Periodic version for spectral analysis.
    blackman_window : Similar 3-term window with less suppression.
    """
    return torch.ops.torchscience.nuttall_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
