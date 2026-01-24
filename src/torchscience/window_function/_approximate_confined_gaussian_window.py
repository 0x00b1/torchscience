from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def approximate_confined_gaussian_window(
    n: int,
    sigma: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Approximate confined Gaussian window function (symmetric).

    Computes a symmetric approximate confined Gaussian window of length n.
    This window approximates the optimal confined Gaussian window, which
    minimizes the product of effective duration and effective bandwidth
    (time-frequency uncertainty) while being confined to a finite interval.

    The approximate version provides a computationally simpler form that
    forces the window to zero at the endpoints while preserving the
    Gaussian shape.

    Mathematical Definition
    -----------------------
    The approximate confined Gaussian window is defined as:

        w[k] = (G[k] - G[0]) / (1 - G[0])

    where G[k] = exp(-0.5 * ((k - center) / (sigma * center))^2)
    is the standard Gaussian window, center = (n-1)/2, and G[0] is the
    Gaussian value at the endpoints.

    This simplifies to:

        w[k] = (exp(-0.5 * ((k - center) / (sigma * center))^2) - exp(-0.5 / sigma^2))
               / (1 - exp(-0.5 / sigma^2))

    Properties
    ----------
    - Forces window values to exactly zero at both endpoints
    - Preserves the smooth bell-shaped Gaussian characteristic
    - Maximum value of 1.0 at the center
    - Better spectral leakage properties than standard Gaussian near boundaries
    - sigma controls the trade-off between time and frequency resolution

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    sigma : float or Tensor
        Standard deviation parameter controlling the window width.
        Larger values produce wider windows with better frequency resolution.
        Smaller values produce narrower windows with better time resolution.
        Typical values are in the range [0.3, 0.5].
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
    This function supports autograd - gradients flow through the sigma
    parameter.

    The confined Gaussian window family aims to achieve optimal
    time-frequency localization as described by the uncertainty principle.
    The approximate version trades some optimality for computational
    simplicity while maintaining the key property of zero-valued endpoints.

    See Also
    --------
    gaussian_window : Standard Gaussian window without confinement.
    confined_gaussian_window : Optimal confined Gaussian (more complex).
    periodic_approximate_confined_gaussian_window : Periodic version.
    """
    if n < 0:
        raise ValueError(
            f"approximate_confined_gaussian_window: n must be non-negative, got {n}"
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

    return torch.ops.torchscience.approximate_confined_gaussian_window(
        n, sigma, dtype, layout, device
    )
