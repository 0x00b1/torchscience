from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def periodic_triangular_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Triangular window function (periodic).

    Computes a periodic triangular window of length n, suitable for use in
    spectral analysis with the FFT. Unlike the Bartlett window which has
    zero endpoints, the triangular window has non-zero endpoints.

    Mathematical Definition
    -----------------------
    The periodic triangular window is computed as the first n points of an
    (n+1)-point symmetric triangular window.

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
    triangular_window : Symmetric version for filter design.
    periodic_bartlett_window : Similar but with zero endpoints.
    """
    return torch.ops.torchscience.periodic_triangular_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
