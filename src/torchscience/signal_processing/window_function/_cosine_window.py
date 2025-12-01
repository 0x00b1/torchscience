from typing import Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def cosine_window(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Cosine window function (symmetric).

    Computes a symmetric cosine (sine) window of length n. The cosine window
    is a simple half-period of a sine wave, providing smooth tapering with
    moderate side lobe suppression.

    Mathematical Definition
    -----------------------
    The symmetric cosine window is defined as:

        w[k] = sin(pi * k / (n - 1)),  for k = 0, 1, ..., n-1

    Properties
    ----------
    - Smooth half-sine shape
    - Zero at endpoints
    - Moderate side lobe suppression

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
    periodic_cosine_window : Periodic version for spectral analysis.
    general_cosine_window : Generalized sum-of-cosines window.
    """
    return torch.ops.torchscience.cosine_window(
        n,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
