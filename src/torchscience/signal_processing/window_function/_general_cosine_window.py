from typing import Optional, Sequence, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def general_cosine_window(
    n: int,
    coeffs: Union[Sequence[float], Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    General cosine window function (symmetric).

    Computes a symmetric general cosine window of length n. The general cosine
    window is a sum-of-cosines form that encompasses many common windows
    including Hann, Hamming, Blackman, and Nuttall.

    Mathematical Definition
    -----------------------
    The symmetric general cosine window is defined as:

        w[k] = sum_{j=0}^{M-1} coeffs[j] * (-1)^j * cos(2 * pi * j * k / (n - 1))

    for k = 0, 1, ..., n-1, where M is the number of coefficients.

    Special Cases
    -------------
    - coeffs = [0.5, 0.5]: Hann window
    - coeffs = [0.54, 0.46]: Hamming window
    - coeffs = [0.42, 0.5, 0.08]: Blackman window
    - coeffs = [0.355768, 0.487396, 0.144232, 0.012604]: Nuttall window

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    coeffs : sequence of float or Tensor
        The cosine series coefficients. Must be a 1-D tensor or sequence.
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
    This function supports autograd - gradients flow through the coeffs parameter.

    See Also
    --------
    periodic_general_cosine_window : Periodic version for spectral analysis.
    blackman_window : Blackman window as special case.
    nuttall_window : Nuttall window as special case.
    """
    if not isinstance(coeffs, Tensor):
        coeffs = torch.tensor(
            coeffs, dtype=dtype or torch.float32, device=device
        )

    return torch.ops.torchscience.general_cosine_window(
        n,
        coeffs,
        dtype=dtype,
        layout=layout,
        device=device,
    )
