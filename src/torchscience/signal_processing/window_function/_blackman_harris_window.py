from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def blackman_harris_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Blackman-Harris window function (symmetric).

    Computes a symmetric Blackman-Harris window of length n. The Blackman-Harris
    window is a 4-term generalized cosine window with coefficients optimized
    for minimal side lobe levels.

    Mathematical Definition
    -----------------------
    The symmetric Blackman-Harris window is defined as:

    w[k] = a0 - a1*cos(2*pi*k/(n-1)) + a2*cos(4*pi*k/(n-1)) - a3*cos(6*pi*k/(n-1))

    where:
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168

    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - 4-term generalized cosine window
    - Very low side lobe level: approximately -92 dB
    - Wider main lobe compared to simpler windows
    - Used when side lobe suppression is critical
    - Coefficients sum to 1 at center: a0 + a1 + a2 + a3 = 1

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
    periodic_blackman_harris_window : Periodic version for spectral analysis with FFT.
    nuttall_window : Similar 4-term window with different coefficients.
    blackman_window : Simpler 3-term window.
    """
    return torch.ops.torchscience.blackman_harris_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
