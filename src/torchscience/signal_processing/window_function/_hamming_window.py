from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def hamming_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Hamming window function (symmetric).

    Computes a symmetric Hamming window of length n. The Hamming window is
    optimized to minimize the nearest side lobe, providing better frequency
    resolution than the Hann window at the expense of slower side lobe decay.

    Mathematical Definition
    -----------------------
    The symmetric Hamming window is defined as:

        w[k] = 0.54 - 0.46 * cos(2 * pi * k / (n - 1)),  for k = 0, 1, ..., n-1

    Properties
    ----------
    - Main lobe width: 8*pi/n
    - Side lobe level: -42.7 dB (first side lobe)
    - Non-zero at endpoints (approximately 0.08)

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
    periodic_hamming_window : Periodic version for spectral analysis.
    general_hamming_window : Generalized form with adjustable alpha.
    """
    return torch.ops.torchscience.hamming_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
