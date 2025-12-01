from __future__ import annotations

from typing import Tuple

import torch

from torchscience.differentiation._finite_difference_stencil import (
    finite_difference_stencil,
)
from torchscience.differentiation._stencil import FiniteDifferenceStencil


def gradient_stencils(
    ndim: int,
    accuracy: int = 2,
    kind: str = "central",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tuple[FiniteDifferenceStencil, ...]:
    """Generate tuple of gradient stencils, one per dimension.

        The gradient is a vector of first partial derivatives:
        grad f = (df/dx, df/dy, ...)

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions.
        accuracy : int
            Accuracy order of the approximation. Default is 2.
        kind : str
            Stencil type: "central", "forward", or "backward".
        dtype : torch.dtype, optional
            Output dtype. Default is torch.float64.
        device : torch.device, optional
            Output device. Default is CPU.

        Returns
        -------
        Tuple[FiniteDifferenceStencil, ...]
            Tuple of n stencils, one for each dimension.
            stencils[i].derivative has 1 in position i and 0 elsewhere.

        Examples
        --------
        >>> stencils = gradient_stencils(ndim=2, accuracy=2)
        >>> len(stencils)
        2
    import torchscience.differentiation._derivative    >>> torchscience.differentiation._derivative.derivative
        (1, 0)
    import torchscience.differentiation._derivative    >>> torchscience.differentiation._derivative.derivative
        (0, 1)
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")

    stencils = []
    for dim in range(ndim):
        # Create derivative tuple with 1 in position dim
        derivative = tuple(1 if d == dim else 0 for d in range(ndim))
        stencil = finite_difference_stencil(
            derivative=derivative,
            accuracy=accuracy,
            kind=kind,
            dtype=dtype,
            device=device,
        )
        stencils.append(stencil)

    return tuple(stencils)
