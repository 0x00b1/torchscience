from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def general_hamming_window(
    n: int,
    alpha: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    General Hamming window function (symmetric).

    Computes a symmetric generalized Hamming window of length n. The general
    Hamming window family includes the standard Hamming (alpha=0.54) and
    Hann (alpha=0.5) windows as special cases.

    Mathematical Definition
    -----------------------
    The symmetric general Hamming window is defined as:

        w[k] = alpha - (1 - alpha) * cos(2 * pi * k / (n - 1))

    for k = 0, 1, ..., n-1.

    Special Cases
    -------------
    - alpha = 0.5: Hann window
    - alpha = 0.54: Standard Hamming window

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
    periodic_general_hamming_window : Periodic version for spectral analysis.
    hamming_window : Standard Hamming window (alpha=0.54).
    hann_window : Hann window (equivalent to alpha=0.5).
    """
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(
            alpha, dtype=dtype or torch.float32, device=device
        )

    return torch.ops.torchscience.general_hamming_window(
        n,
        alpha,
        dtype=dtype,
        layout=layout,
        device=device,
    )
