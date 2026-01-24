from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def welch_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Welch window function (symmetric).

    Computes a symmetric Welch window of length n. The Welch window is a
    parabolic window with good frequency resolution and moderate side lobe
    levels.

    Mathematical Definition
    -----------------------
    The symmetric Welch window is defined as:

        w[k] = 1 - ((k - center) / center)^2,  for k = 0, 1, ..., n-1

    where center = (n-1)/2.

    Properties
    ----------
    - Parabolic shape
    - Continuous first derivative at endpoints
    - Side lobe level: -21.3 dB

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
    periodic_welch_window : Periodic version for spectral analysis with FFT.
    """
    return torch.ops.torchscience.welch_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
