from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def flat_top_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Flat-top window function (symmetric).

    Computes a symmetric flat-top window of length n. The flat-top window is a
    5-term generalized cosine window optimized for accurate amplitude measurements
    in the frequency domain with minimal scalloping error.

    Mathematical Definition
    -----------------------
    The symmetric flat-top window is defined as:

    w[k] = a0 - a1*cos(2*pi*k/(n-1)) + a2*cos(4*pi*k/(n-1))
           - a3*cos(6*pi*k/(n-1)) + a4*cos(8*pi*k/(n-1))

    where:
        a0 = 0.21557895
        a1 = 0.41663158
        a2 = 0.277263158
        a3 = 0.083578947
        a4 = 0.006947368

    for k = 0, 1, ..., n-1.

    Properties
    ----------
    - 5-term generalized cosine window
    - Very flat passband (less than 0.01 dB ripple)
    - Used for accurate amplitude measurements in spectral analysis
    - Can have negative values (not bounded to [0, 1])
    - Coefficients match scipy.signal.windows.flattop

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    dtype : torch.dtype, optional
        The desired data type of the returned tensor.
    layout : torch.layout, optional
        The desired layout of the returned tensor.
    device : torch.device, optional
        The desired device of the returned tensor.
    requires_grad : bool, optional
        If True, the returned tensor will require gradients.

    Returns
    -------
    Tensor
        A 1-D tensor of size (n,) containing the window values.

    See Also
    --------
    periodic_flat_top_window : Periodic version for spectral analysis with FFT.
    blackman_harris_window : 4-term window with low side lobes.
    nuttall_window : Similar 4-term window.

    References
    ----------
    .. [1] D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for
           Measurement Systems", Springer Media, 2006, p. 70
    """
    return torch.ops.torchscience.flat_top_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
