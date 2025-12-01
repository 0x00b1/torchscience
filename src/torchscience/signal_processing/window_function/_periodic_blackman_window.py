from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_blackman_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Blackman window function (periodic).

    Computes a periodic Blackman window of length n, suitable for use in
    spectral analysis with the FFT.

    Mathematical Definition
    -----------------------
    The periodic Blackman window is defined as:

        w[k] = 0.42 - 0.5 * cos(2 * pi * k / n) + 0.08 * cos(4 * pi * k / n)

    for k = 0, 1, ..., n-1. The denominator is n (not n-1).

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
    blackman_window : Symmetric version for filter design.
    """
    return torch.ops.torchscience.periodic_blackman_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
