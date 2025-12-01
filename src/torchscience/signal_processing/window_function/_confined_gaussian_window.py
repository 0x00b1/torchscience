from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def confined_gaussian_window(
    n: int,
    sigma: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Confined Gaussian window function (symmetric).

    Computes a symmetric confined Gaussian window of length n. The confined
    Gaussian window is a modification of the standard Gaussian window that
    is designed to be exactly zero at its boundaries, while maintaining the
    smoothness properties of the Gaussian.

    Mathematical Definition
    -----------------------
    The symmetric confined Gaussian window is defined as:

        w[k] = G(t_k) - G(0.5) * (G(t_k - 1) + G(t_k + 1)) / (G(0.5) + G(1.5))

    where:
        - t_k = (k - (n-1)/2) / (n-1) is the normalized position in [-0.5, 0.5]
        - G(t) = exp(-0.5 * (t / sigma)^2) is the Gaussian function

    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - Exactly zero at the boundaries (w[0] = w[n-1] = 0)
    - Smooth bell-shaped curve
    - Better frequency localization than standard Gaussian due to compact support
    - Maintains continuity and smoothness at boundaries

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    sigma : float or Tensor
        Standard deviation parameter controlling the window width.
        Larger values produce wider windows. Typical values range from 0.1 to 0.5.
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
    This function supports autograd - gradients flow through the sigma parameter.

    References
    ----------
    .. [1] S. Gröchenig and H. Feichtinger, "Gabor frames and time-frequency
           analysis of distributions," Journal of Functional Analysis, 1997.

    See Also
    --------
    gaussian_window : Standard Gaussian window (non-zero at boundaries).
    approximate_confined_gaussian_window : Faster approximation.
    periodic_confined_gaussian_window : Periodic version for spectral analysis.
    """
    if n < 0:
        raise ValueError(
            f"confined_gaussian_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    if not isinstance(sigma, Tensor):
        target_dtype = dtype or torch.float32
        sigma = torch.tensor(sigma, dtype=target_dtype, device=device)

    return torch.ops.torchscience.confined_gaussian_window(
        n, sigma, dtype, layout, device
    )
