from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_flat_top_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Flat-top window function (periodic).

    Computes a periodic flat-top window of length n, suitable for use in
    spectral analysis with the FFT. The periodic version is equivalent to
    taking the first n points of an (n+1)-length symmetric flat-top window.

    The flat-top window is a 5-term generalized cosine window optimized for
    accurate amplitude measurements in the frequency domain with minimal
    scalloping error. It has a very flat passband (less than 0.01 dB ripple).

    Note that the flat-top window can have negative values and is not bounded
    to the [0, 1] range.

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
    flat_top_window : Symmetric version for filter design.
    periodic_blackman_harris_window : Similar 4-term periodic window.
    """
    return torch.ops.torchscience.periodic_flat_top_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
