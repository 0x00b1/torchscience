from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def tukey_window(
    n: int,
    alpha: Union[float, Tensor],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Tukey (tapered cosine) window function (symmetric).

    Computes a symmetric Tukey window of length n. The Tukey window is a
    cosine-tapered window that combines properties of a rectangular window
    (flat top) with a cosine window (smooth edges).

    Mathematical Definition
    -----------------------
    The symmetric Tukey window is defined as:

    For 0 <= k < alpha * N / 2:
        w[k] = 0.5 * (1 - cos(2 * pi * k / (alpha * N)))

    For alpha * N / 2 <= k < N * (1 - alpha / 2):
        w[k] = 1

    For N * (1 - alpha / 2) <= k < N:
        w[k] = 0.5 * (1 + cos(2 * pi * (k - N * (1 - alpha / 2)) / (alpha * N)))

    where N is the window denominator (n-1 for symmetric windows).

    Properties
    ----------
    - alpha = 0: rectangular window (all ones)
    - alpha = 1: Hann window
    - alpha in (0, 1): cosine-tapered window with flat top
    - Smooth transitions at the edges controlled by alpha

    Parameters
    ----------
    n : int
        Number of points in the output window. Must be non-negative.
    alpha : float or Tensor
        Shape parameter controlling the fraction of the window inside the
        tapered region. Must be in [0, 1].
        - alpha = 0: rectangular window
        - alpha = 1: Hann window
        - alpha in (0, 1): taper width is alpha * n / 2 on each side
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
    periodic_tukey_window : Periodic version for spectral analysis.
    hann_window : Equivalent to tukey_window with alpha=1.
    """
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(
            alpha, dtype=dtype or torch.float32, device=device
        )

    return torch.ops.torchscience.tukey_window(
        n,
        alpha,
        dtype=dtype,
        layout=layout,
        device=device,
    )
