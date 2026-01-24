from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def gaussian_window(
    n: int,
    std: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Gaussian window function (symmetric).

    Computes a symmetric Gaussian window of length n. The Gaussian window
    provides excellent time-frequency localization and is widely used in
    time-frequency analysis and filter design.

    Mathematical Definition
    -----------------------
    The symmetric Gaussian window is defined as:

        w[k] = exp(-0.5 * ((k - center) / (std * center))^2)

    for k = 0, 1, ..., n-1, where center = (n-1)/2.

    Properties
    ----------
    - Smooth bell-shaped curve
    - No side lobes in theory (Gaussian is its own Fourier transform)
    - Trade-off between time and frequency resolution controlled by std

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    std : float or Tensor
        Standard deviation parameter controlling the window width.
        Larger values produce wider windows with better frequency resolution.
        Smaller values produce narrower windows with better time resolution.
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
    This function supports autograd - gradients flow through the std parameter.

    See Also
    --------
    periodic_gaussian_window : Periodic version for spectral analysis.
    """
    if not isinstance(std, Tensor):
        std = torch.tensor(std, dtype=dtype or torch.float32, device=device)

    return torch.ops.torchscience.gaussian_window(
        n,
        std,
        dtype=dtype,
        layout=layout,
        device=device,
    )
