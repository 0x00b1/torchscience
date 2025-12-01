from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_dolph_chebyshev_window(
    n: int,
    attenuation: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Dolph-Chebyshev window function (periodic).

    Computes a periodic Dolph-Chebyshev window of length n. The periodic version
    is designed for spectral analysis where the window will be used with DFT/FFT.

    The periodic window is computed by generating a symmetric window of length
    n+1 and returning the first n points, ensuring proper periodicity when the
    window is repeated.

    Properties
    ----------
    - All sidelobes have equal height (equiripple)
    - Designed for spectral analysis with FFT
    - Periodic: w[n] would equal w[0] if computed

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    attenuation : float or Tensor
        Desired sidelobe attenuation in decibels (dB). Must be positive.
        Common values:
        - 50 dB: moderate sidelobe suppression
        - 60 dB: good sidelobe suppression
        - 80 dB: excellent sidelobe suppression
        - 100 dB: very high sidelobe suppression
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
    This function supports autograd - gradients flow through the attenuation
    parameter.

    See Also
    --------
    dolph_chebyshev_window : Symmetric version.
    """
    if n < 0:
        raise ValueError(
            f"periodic_dolph_chebyshev_window: n must be non-negative, got {n}"
        )

    if n == 0:
        target_dtype = dtype or torch.float32
        return torch.empty(0, dtype=target_dtype, layout=layout, device=device)

    if n == 1:
        target_dtype = dtype or torch.float32
        return torch.ones(1, dtype=target_dtype, layout=layout, device=device)

    target_dtype = dtype or torch.float32

    if not isinstance(attenuation, Tensor):
        attenuation = torch.tensor(
            attenuation, dtype=target_dtype, device=device
        )

    # Validate attenuation > 0
    if attenuation.item() <= 0:
        raise ValueError(
            f"periodic_dolph_chebyshev_window: attenuation must be positive, got {attenuation.item()}"
        )

    return torch.ops.torchscience.periodic_dolph_chebyshev_window(
        n, attenuation, dtype, layout, device
    )
