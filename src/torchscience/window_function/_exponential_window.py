from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def exponential_window(
    n: int,
    tau: Union[float, Tensor] = 1.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Exponential (Poisson) window function (symmetric).

    Computes a symmetric exponential window of length n. The exponential
    window provides a smooth decay from the center to the edges.

    Mathematical Definition
    -----------------------
    The symmetric exponential window is defined as:

        w[k] = exp(-|k - center| / tau)

    for k = 0, 1, ..., n-1, where center = (n-1)/2.

    Properties
    ----------
    - Symmetric exponential decay from center
    - tau controls the decay rate (larger tau = slower decay)
    - Maximum value of 1.0 at the center

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    tau : float or Tensor, optional
        Decay parameter controlling the window shape. Default is 1.0.
        Larger values produce slower decay (wider window).
        For center=0, use tau = -(n-1) / ln(x) where x is the fraction
        remaining at the end.
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
    This function supports autograd - gradients flow through the tau parameter.

    See Also
    --------
    periodic_exponential_window : Periodic version for spectral analysis.
    """
    if n < 0:
        raise ValueError(
            f"exponential_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    if not isinstance(tau, Tensor):
        target_dtype = dtype or torch.float32
        tau = torch.tensor(tau, dtype=target_dtype, device=device)

    return torch.ops.torchscience.exponential_window(
        n, tau, dtype, layout, device
    )
