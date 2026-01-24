from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def kaiser_window(
    n: int,
    beta: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Kaiser window function (symmetric).

    Computes a symmetric Kaiser window of length n. The Kaiser window is
    an approximation to the DPSS (discrete prolate spheroidal sequence)
    window, which provides optimal energy concentration.

    Mathematical Definition
    -----------------------
    The symmetric Kaiser window is defined as:

        w[k] = I_0(beta * sqrt(1 - ((k - (n-1)/2) / ((n-1)/2))^2)) / I_0(beta)

    for k = 0, 1, ..., n-1, where I_0 is the modified Bessel function
    of the first kind, order 0.

    Properties
    ----------
    - Provides excellent sidelobe suppression
    - beta controls the trade-off between main lobe width and sidelobe level
    - Larger beta: narrower main lobe, higher sidelobes
    - Smaller beta: wider main lobe, lower sidelobes
    - beta = 0: rectangular window

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    beta : float or Tensor
        Shape parameter controlling the window shape.
        Common values:
        - beta = 0: rectangular window
        - beta = 5: similar to Hamming window
        - beta = 6: similar to Hann window
        - beta = 8.6: similar to Blackman window
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values.

    Notes
    -----
    This function supports autograd - gradients flow through the beta parameter.

    See Also
    --------
    periodic_kaiser_window : Periodic version for spectral analysis.
    """
    if n < 0:
        raise ValueError(f"kaiser_window: n must be non-negative, got {n}")

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    target_dtype = dtype or torch.float32

    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=target_dtype, device=device)

    return torch.ops.torchscience.kaiser_window(n, beta, dtype, layout, device)
