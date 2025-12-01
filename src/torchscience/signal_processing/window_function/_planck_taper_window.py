from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def planck_taper_window(
    n: int,
    epsilon: Union[float, Tensor] = 0.1,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Planck-taper window function (symmetric).

    Computes a symmetric Planck-taper window of length n. The Planck-taper
    window provides smooth transitions using a sigmoid function, creating
    infinitely differentiable tapers while maintaining a flat central region.

    Mathematical Definition
    -----------------------
    The symmetric Planck-taper window is defined for normalized position
    t = k / (n - 1) where k = 0, 1, ..., n-1:

    For 0 < t < epsilon:
        w(t) = 1 / (1 + exp(Z⁺(t)))
        where Z⁺(t) = epsilon * (1/t + 1/(t - epsilon))

    For epsilon <= t <= 1 - epsilon:
        w(t) = 1

    For 1 - epsilon < t < 1:
        w(t) = 1 / (1 + exp(Z⁻(t)))
        where Z⁻(t) = epsilon * (1/(1 - t) + 1/(1 - t - epsilon))

    At the boundaries t = 0 and t = 1:
        w(t) = 0

    Properties
    ----------
    - Infinitely differentiable (C∞) at all points
    - epsilon controls the fraction of the window that is tapered on each side
    - epsilon = 0: rectangular window (all ones, discontinuous at edges)
    - epsilon = 0.5: fully tapered (no flat region)
    - The window uses a smooth sigmoid transition based on the Planck function
    - Symmetric about the center

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    epsilon : float or Tensor, optional
        Taper parameter controlling the fraction of the window inside the
        tapered region on each side. Must be in [0, 0.5]. Default is 0.1.
        - epsilon = 0: rectangular window
        - epsilon = 0.5: fully tapered window
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
    This function supports autograd - gradients flow through the epsilon
    parameter.

    The Planck-taper window is named after the physicist Max Planck and uses
    a similar functional form to the Planck distribution. It provides an
    alternative to the Tukey window with smoother (infinitely differentiable)
    transitions.

    See Also
    --------
    periodic_planck_taper_window : Periodic version for spectral analysis.
    tukey_window : Cosine-tapered window with similar flat-top structure.
    """
    if n < 0:
        raise ValueError(
            f"planck_taper_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    target_dtype = dtype or torch.float32

    if not isinstance(epsilon, Tensor):
        epsilon = torch.tensor(epsilon, dtype=target_dtype, device=device)

    return torch.ops.torchscience.planck_taper_window(
        n, epsilon, dtype, layout, device
    )
