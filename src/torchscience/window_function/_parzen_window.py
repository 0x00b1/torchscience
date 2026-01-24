from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def parzen_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Parzen window function (symmetric).

    Computes a symmetric Parzen window of length n. The Parzen window is also
    known as the de la Vallee Poussin window and is a 4th-order B-spline
    window with a piecewise cubic polynomial form.

    Mathematical Definition
    -----------------------
    The symmetric Parzen window is defined as a piecewise function:

    For |k - center| <= (n-1)/4:
        w[k] = 1 - 6*x^2 + 6*|x|^3

    For (n-1)/4 < |k - center| <= n/2:
        w[k] = 2*(1 - |x|)^3

    where x = (k - center) / (n/2) and center = (n-1)/2.

    Properties
    ----------
    - Piecewise cubic polynomial (4th-order B-spline)
    - Smooth transitions with continuous derivatives
    - Good spectral properties with relatively low side lobes
    - Main lobe width: 8*pi/n
    - Maximum side lobe level: approximately -53.1 dB

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
    periodic_parzen_window : Periodic version for spectral analysis with FFT.
    triangular_window : Linear (first-order) window.
    """
    return torch.ops.torchscience.parzen_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
