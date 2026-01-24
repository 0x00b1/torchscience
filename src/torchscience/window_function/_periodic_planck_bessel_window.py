from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_planck_bessel_window(
    n: int,
    epsilon: Union[float, Tensor] = 0.1,
    beta: Union[float, Tensor] = 8.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Planck-Bessel window function (periodic).

    Computes a periodic Planck-Bessel window of length n. The periodic version
    is designed for spectral analysis where the window will be used with DFT/FFT.

    Mathematical Definition
    -----------------------
    The periodic Planck-Bessel window uses a denominator of n (instead of n-1)
    for proper periodicity:

        w[k] = planck_taper[k] * kaiser[k]

    where:
        planck_taper[k] uses the Planck function with denominator n:
            - For k in (0, epsilon * n):
                z = epsilon * n * (1/k + 1/(k - epsilon*n))
                planck_taper[k] = 1 / (1 + exp(z))
            - For k in [epsilon * n, (1-epsilon) * n]:
                planck_taper[k] = 1
            - For k in ((1-epsilon) * n, n):
                planck_taper[k] = planck_taper[n-k]  (mirrored from left)

        kaiser[k] = I_0(beta * sqrt(1 - ((k - n/2) / (n/2))^2)) / I_0(beta)
            where I_0 is the modified Bessel function of the first kind, order 0.

    Properties
    ----------
    - Combines smooth Planck taper transitions with Kaiser-Bessel sidelobe control
    - epsilon controls the width of the taper regions at the edges
    - beta controls the Kaiser-Bessel shape (sidelobe suppression)
    - Designed for spectral analysis with FFT

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    epsilon : float or Tensor, optional
        Planck taper parameter controlling the width of the taper regions.
        Must be in [0, 0.5]. Default is 0.1.
        - epsilon = 0: no Planck taper (reduces to periodic Kaiser window)
        - epsilon = 0.5: taper extends to the center
    beta : float or Tensor, optional
        Kaiser-Bessel shape parameter. Default is 8.0.
        Common values:
        - beta = 0: rectangular window (no Bessel shaping)
        - beta = 5: similar to Hamming window
        - beta = 6: similar to Hann window
        - beta = 8.6: similar to Blackman window
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
    This function supports autograd - gradients flow through both the
    epsilon and beta parameters.

    See Also
    --------
    planck_bessel_window : Symmetric version.
    periodic_kaiser_window : Pure periodic Kaiser-Bessel window.
    """
    if n < 0:
        raise ValueError(
            f"periodic_planck_bessel_window: n must be non-negative, got {n}"
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
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=epsilon.dtype, device=epsilon.device)

    return torch.ops.torchscience.periodic_planck_bessel_window(
        n, epsilon, beta, dtype, layout, device
    )
