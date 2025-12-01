from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation.__generate_1d_offsets import (
    _generate_1d_offsets,
)
from torchscience.differentiation.__generate_nd_stencil import (
    _generate_nd_stencil,
)
from torchscience.differentiation._fornberg_weights import fornberg_weights
from torchscience.differentiation._stencil import FiniteDifferenceStencil


def finite_difference_stencil(
    derivative: Union[int, Tuple[int, ...]],
    accuracy: int = 2,
    kind: str = "central",
    offsets: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> FiniteDifferenceStencil:
    """Generate finite difference stencil for arbitrary derivative.

    Uses Fornberg's algorithm to compute optimal coefficients for the
    given grid points and derivative order.

    Parameters
    ----------
    derivative : int or tuple of int
        Derivative order. For 1D, can be an int. For n-D, use tuple like
        (2, 0) for d^2/dx^2 or (1, 1) for mixed partial d^2/dxdy.
    accuracy : int
        Accuracy order of the approximation. Error is O(dx^accuracy).
        Must be positive. Default is 2 (second-order accuracy).
    kind : str
        Stencil type: "central", "forward", or "backward".
        - "central": symmetric around the point (most accurate per point)
        - "forward": uses points at and ahead of the point
        - "backward": uses points at and behind the point
    offsets : Tensor, optional
        Custom grid point offsets. If provided, overrides `kind`.
        Shape should be (n_points,) for 1D or (n_points, ndim) for n-D.
    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64.
    device : torch.device, optional
        Output device. Default is CPU.

    Returns
    -------
    FiniteDifferenceStencil
        Stencil with offsets and coefficients.

    Raises
    ------
    ValueError
        If derivative <= 0, accuracy <= 0, or invalid kind.

    Examples
    --------
    >>> stencil = finite_difference_stencil(derivative=1, accuracy=2)
    >>> stencil.coeffs  # Central first derivative: [-0.5, 0, 0.5]

    >>> stencil = finite_difference_stencil(derivative=2, accuracy=4)
    >>> stencil.coeffs  # Central second derivative, 4th order accuracy

    >>> stencil = finite_difference_stencil(derivative=(1, 1), accuracy=2)
    >>> stencil.coeffs  # Mixed partial d^2/dxdy
    """
    # Normalize derivative to tuple
    if isinstance(derivative, int):
        derivative_tuple = (derivative,)
    else:
        derivative_tuple = tuple(derivative)

    ndim = len(derivative_tuple)

    # Validation
    if any(d < 0 for d in derivative_tuple):
        raise ValueError("derivative orders must be non-negative")
    if sum(derivative_tuple) == 0:
        raise ValueError("total derivative order must be positive")
    if accuracy <= 0:
        raise ValueError("accuracy must be positive")
    if kind not in ("central", "forward", "backward"):
        raise ValueError(
            f"kind must be 'central', 'forward', or 'backward', got '{kind}'"
        )

    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")

    # Generate offsets if not provided
    if offsets is not None:
        # Custom offsets provided
        if offsets.dim() == 1:
            offsets = offsets.unsqueeze(-1)  # (n,) -> (n, 1)
        custom_offsets = offsets.to(device=device)
    else:
        # Generate standard offsets based on kind and accuracy
        custom_offsets = None

    if ndim == 1:
        # 1D case
        deriv_order = derivative_tuple[0]

        if custom_offsets is not None:
            grid_offsets = custom_offsets.squeeze(-1)
        else:
            grid_offsets = _generate_1d_offsets(
                deriv_order, accuracy, kind, device
            )

        # Compute weights using Fornberg's algorithm
        weights = fornberg_weights(deriv_order, grid_offsets)

        return FiniteDifferenceStencil(
            offsets=grid_offsets.unsqueeze(-1).long(),
            coeffs=weights.to(dtype),
            derivative=derivative_tuple,
            accuracy=accuracy,
        )
    else:
        # Multi-dimensional case: tensor product of 1D stencils
        return _generate_nd_stencil(
            derivative_tuple, accuracy, kind, dtype, device
        )
