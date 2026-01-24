from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def bartlett_hann_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Bartlett-Hann window function (symmetric).

    Computes a symmetric Bartlett-Hann window of length n. The Bartlett-Hann
    window is a combination of the Bartlett (triangular) and Hann windows,
    providing a smooth transition between zero endpoints while maintaining
    good spectral characteristics.

    Mathematical Definition
    -----------------------
    The symmetric Bartlett-Hann window is defined as:

    w[k] = a0 - a1*|k/(n-1) - 0.5| - a2*cos(2*pi*k/(n-1))

    where:
        a0 = 0.62
        a1 = 0.48
        a2 = 0.38

    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - Combination of Bartlett and Hann window characteristics
    - Smooth tapering with zero endpoints for symmetric version
    - Better side lobe suppression than pure Bartlett window
    - More gradual roll-off than pure Hann window
    - Coefficients optimized for good frequency response

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
    periodic_bartlett_hann_window : Periodic version for spectral analysis with FFT.
    bartlett_window : Triangular window (Bartlett).
    hann_window : Hann window.
    """
    return torch.ops.torchscience.bartlett_hann_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
