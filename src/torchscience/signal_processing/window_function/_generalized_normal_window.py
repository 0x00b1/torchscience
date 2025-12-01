from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def generalized_normal_window(
    n: int,
    p: Union[float, Tensor],
    sigma: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Generalized normal (generalized Gaussian) window function (symmetric).

    Computes a symmetric generalized normal window of length n. This window
    generalizes the Gaussian window by allowing control over the shape via
    the exponent parameter p.

    Mathematical Definition
    -----------------------
    The symmetric generalized normal window is defined as:

        w[k] = exp(-|(k - center) / sigma|^p)

    for k = 0, 1, ..., n-1, where center = (n-1)/2.

    Properties
    ----------
    - p = 2: Standard Gaussian window
    - p = 1: Laplacian (double exponential) window
    - p < 2: Heavier tails than Gaussian (more peaked)
    - p > 2: Lighter tails than Gaussian (flatter top, approaches rectangular)
    - p -> infinity: Approaches rectangular window
    - sigma controls the effective width of the window

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    p : float or Tensor
        Shape parameter controlling the window shape.
        Must be positive. Common values:
        - p = 1: Laplacian/exponential decay
        - p = 2: Gaussian window
        - p > 2: Flatter top, steeper sides
    sigma : float or Tensor
        Scale parameter controlling the window width.
        Larger values produce wider windows.
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
    This function supports autograd - gradients flow through both p and sigma
    parameters.

    See Also
    --------
    gaussian_window : Special case with p = 2.
    exponential_window : Related window with exponential decay.
    """
    if n < 0:
        raise ValueError(
            f"generalized_normal_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    target_dtype = dtype or torch.float32

    if not isinstance(p, Tensor):
        p = torch.tensor(p, dtype=target_dtype, device=device)

    if not isinstance(sigma, Tensor):
        sigma = torch.tensor(sigma, dtype=target_dtype, device=device)

    # Ensure p and sigma have compatible dtypes
    if p.dtype != sigma.dtype:
        common_dtype = torch.promote_types(p.dtype, sigma.dtype)
        p = p.to(dtype=common_dtype)
        sigma = sigma.to(dtype=common_dtype)

    return torch.ops.torchscience.generalized_normal_window(
        n, p, sigma, dtype, layout, device
    )
