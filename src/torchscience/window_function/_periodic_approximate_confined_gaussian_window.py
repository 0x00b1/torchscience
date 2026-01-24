from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_approximate_confined_gaussian_window(
    n: int,
    sigma: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Approximate confined Gaussian window function (periodic).

    Computes a periodic approximate confined Gaussian window of length n. This
    is a computationally simpler approximation of the confined Gaussian window
    that still ensures the window goes to zero at the boundaries. The periodic
    version is designed for spectral analysis where the window will be used
    with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic approximate confined Gaussian window is defined as:

        w[k] = G(t_k) - G(-0.5)

    where:
        - t_k = (k - n/2) / n is the normalized position in [-0.5, 0.5)
        - G(t) = exp(-0.5 * (t / sigma)^2) is the Gaussian function

    for k = 0, 1, ..., n-1.

    This approximation subtracts the boundary value of the Gaussian, ensuring
    the window is exactly zero at t = -0.5 and approximately zero at t = 0.5.

    Properties
    ----------
    - Approximately zero at the boundaries
    - Smooth bell-shaped curve similar to Gaussian
    - Computationally simpler than the full confined Gaussian
    - Better frequency localization than standard Gaussian due to compact support
    - Designed for spectral analysis with FFT

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
    .. [1] S. Starosielec and D. Hägele, "Discrete-time windows with minimal
           RMS bandwidth for given RMS temporal width," Signal Processing,
           vol. 102, pp. 240-246, 2014.

    See Also
    --------
    approximate_confined_gaussian_window : Symmetric version.
    periodic_confined_gaussian_window : Full periodic confined Gaussian.
    periodic_gaussian_window : Standard periodic Gaussian window.
    """
    if n < 0:
        raise ValueError(
            f"periodic_approximate_confined_gaussian_window: n must be non-negative, got {n}"
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

    return (
        torch.ops.torchscience.periodic_approximate_confined_gaussian_window(
            n, sigma, dtype, layout, device
        )
    )
