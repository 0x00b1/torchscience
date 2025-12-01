from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def sine_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Sine window function (symmetric).

    Computes a symmetric sine window of length n. The sine window is a simple
    half-period of a sine wave, providing smooth tapering with moderate side
    lobe suppression. This is mathematically identical to the cosine window
    (scipy.signal.windows.cosine).

    Mathematical Definition
    -----------------------
    The symmetric sine window is defined as:

        w[k] = sin(pi * k / (n - 1)),  for k = 0, 1, ..., n-1

    Properties
    ----------
    - Smooth half-sine shape
    - Zero at endpoints
    - Maximum value of 1.0 at the center
    - Moderate side lobe suppression

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
    periodic_sine_window : Periodic version for spectral analysis.
    cosine_window : Equivalent function (same mathematical definition).

    Notes
    -----
    The sine window and cosine window are the same function. Different
    communities use different names for this window. scipy calls it "cosine".
    """
    return torch.ops.torchscience.sine_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
