from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def generalized_adaptive_polynomial_window(
    n: int,
    alpha: Union[float, Tensor] = 2.0,
    beta: Union[float, Tensor] = 1.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Generalized adaptive polynomial window function (symmetric).

    Computes a symmetric generalized adaptive polynomial window of length n.
    This window function provides flexible control over the window shape through
    two parameters that control the polynomial characteristics.

    Mathematical Definition
    -----------------------
    The symmetric generalized adaptive polynomial window is defined as:

        w[k] = (1 - |x[k]|^alpha)^beta

    where x[k] = 2k / (n - 1) - 1 is the normalized position in [-1, 1]
    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - Smooth, bell-shaped window with adjustable characteristics
    - alpha controls the steepness of the polynomial decay
    - beta controls the overall shape/power of the window
    - Window is symmetric about the center
    - Values range from 0 at the edges to 1 at the center

    Special Cases
    -------------
    - alpha=1, beta=1: Triangular (Bartlett-like) window
    - alpha=2, beta=1: Welch window
    - alpha=2, beta=0.5: Similar to cosine window

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    alpha : float or Tensor, optional
        Shape parameter controlling the polynomial degree. Default is 2.0.
        Must be positive. Larger values create steeper transitions near edges.
    beta : float or Tensor, optional
        Shape parameter controlling the power applied to the polynomial.
        Default is 1.0. Must be positive. Larger values create narrower windows.
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
    This function supports autograd - gradients flow through the alpha and
    beta parameters.

    See Also
    --------
    periodic_generalized_adaptive_polynomial_window : Periodic version for
        spectral analysis.
    welch_window : Equivalent to generalized_adaptive_polynomial_window with
        alpha=2, beta=1.
    """
    if n < 0:
        raise ValueError(
            f"generalized_adaptive_polynomial_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    # Convert parameters to tensors
    if not isinstance(alpha, Tensor):
        target_dtype = dtype or torch.float32
        alpha = torch.tensor(alpha, dtype=target_dtype, device=device)

    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=alpha.dtype, device=alpha.device)

    return torch.ops.torchscience.generalized_adaptive_polynomial_window(
        n, alpha, beta, dtype, layout, device
    )
