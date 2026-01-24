from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_general_hamming_window(
    n: int,
    alpha: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    General Hamming window function (periodic).

    Computes a periodic generalized Hamming window of length n, suitable
    for use in spectral analysis with the FFT.

    Mathematical Definition
    -----------------------
    The periodic general Hamming window is defined as:

        w[k] = alpha - (1 - alpha) * cos(2 * pi * k / n)

    for k = 0, 1, ..., n-1. The denominator is n (not n-1).

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    alpha : float or Tensor
        The alpha parameter controlling the window shape.
        Must be in the range [0, 1].
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
    This function supports autograd - gradients flow through the alpha parameter.

    See Also
    --------
    general_hamming_window : Symmetric version for filter design.
    periodic_hamming_window : Standard periodic Hamming window.
    periodic_hann_window : Periodic Hann window (equivalent to alpha=0.5).
    """
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(
            alpha, dtype=dtype or torch.float32, device=device
        )

    return torch.ops.torchscience.periodic_general_hamming_window(
        n,
        alpha,
        dtype=dtype,
        layout=layout,
        device=device,
    )
