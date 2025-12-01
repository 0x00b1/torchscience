from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_hann_poisson_window(
    n: int,
    alpha: Union[float, Tensor] = 1.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Hann-Poisson window function (periodic).

    Computes a periodic Hann-Poisson window of length n. The periodic version
    is designed for spectral analysis where the window will be used with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic Hann-Poisson window is defined as:

        w[k] = 0.5 * (1 - cos(2 * pi * k / n)) * exp(-alpha * |n - 2k| / n)

    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - Combines Hann's smooth tapering with exponential decay
    - alpha controls the exponential decay rate (larger alpha = faster decay)
    - When alpha = 0, reduces to a periodic Hann window
    - Designed for spectral analysis with FFT

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    alpha : float or Tensor, optional
        Exponential decay parameter. Default is 1.0.
        Larger values produce faster decay toward the edges.
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
    This function supports autograd - gradients flow through the alpha parameter.

    See Also
    --------
    hann_poisson_window : Symmetric version.
    periodic_hann_window : Pure periodic Hann window.
    periodic_exponential_window : Pure periodic exponential window.
    """
    if n < 0:
        raise ValueError(
            f"periodic_hann_poisson_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    if not isinstance(alpha, Tensor):
        target_dtype = dtype or torch.float32
        alpha = torch.tensor(alpha, dtype=target_dtype, device=device)

    return torch.ops.torchscience.periodic_hann_poisson_window(
        n, alpha, dtype, layout, device
    )
