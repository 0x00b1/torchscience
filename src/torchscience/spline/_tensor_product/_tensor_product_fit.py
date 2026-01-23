"""Tensor product spline fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._tensor_product_spline import TensorProductSpline


def tensor_product_fit(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    extrapolate: str = "error",
) -> TensorProductSpline:
    """
    Fit a bicubic tensor product spline to gridded data.

    Parameters
    ----------
    x : Tensor
        Grid x-coordinates, shape (nx,). Must be strictly increasing.
    y : Tensor
        Grid y-coordinates, shape (ny,). Must be strictly increasing.
    z : Tensor
        Grid values, shape (nx, ny) or (nx, ny, *value_shape).
    extrapolate : str
        Extrapolation mode.

    Returns
    -------
    spline : TensorProductSpline
        Fitted tensor product spline.

    Notes
    -----
    The tensor product spline stores the grid values directly as coefficients.
    During evaluation, bicubic interpolation is performed using the
    natural cubic spline basis.
    """
    from ._tensor_product_spline import TensorProductSpline

    nx = x.shape[0]
    ny = y.shape[0]

    if nx < 2:
        raise ValueError(f"Need at least 2 x-coordinates, got {nx}")
    if ny < 2:
        raise ValueError(f"Need at least 2 y-coordinates, got {ny}")

    # Check z shape
    if z.shape[0] != nx or z.shape[1] != ny:
        raise ValueError(f"z shape {z.shape} doesn't match grid ({nx}, {ny})")

    # Check strictly increasing
    hx = x[1:] - x[:-1]
    hy = y[1:] - y[:-1]

    if torch.any(hx <= 0):
        raise ValueError("x values must be strictly increasing")
    if torch.any(hy <= 0):
        raise ValueError("y values must be strictly increasing")

    return TensorProductSpline(
        x_knots=x.clone(),
        y_knots=y.clone(),
        coefficients=z.clone(),
        extrapolate=extrapolate,
        batch_size=[],
    )
