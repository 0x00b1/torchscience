"""Tensor product spline partial derivatives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._tensor_product_spline import TensorProductSpline


def tensor_product_derivative(
    spline: TensorProductSpline,
    qx: Tensor,
    qy: Tensor,
    dx: int = 0,
    dy: int = 0,
) -> Tensor:
    """
    Evaluate partial derivative of tensor product spline.

    Parameters
    ----------
    spline : TensorProductSpline
        The tensor product spline.
    qx : Tensor
        Query x-coordinates.
    qy : Tensor
        Query y-coordinates.
    dx : int
        Order of derivative with respect to x.
    dy : int
        Order of derivative with respect to y.

    Returns
    -------
    result : Tensor
        Partial derivative values.
    """
    if dx < 0 or dy < 0:
        raise ValueError("Derivative orders must be non-negative")

    if dx == 0 and dy == 0:
        from ._tensor_product_evaluate import tensor_product_evaluate

        return tensor_product_evaluate(spline, qx, qy)

    x_knots = spline.x_knots
    y_knots = spline.y_knots
    coefficients = spline.coefficients
    extrapolate = spline.extrapolate

    # Handle extrapolation
    x_min, x_max = x_knots[0], x_knots[-1]
    y_min, y_max = y_knots[0], y_knots[-1]

    qx_flat = qx.reshape(-1)
    qy_flat = qy.reshape(-1)

    out_of_bounds_x = (qx_flat < x_min) | (qx_flat > x_max)
    out_of_bounds_y = (qy_flat < y_min) | (qy_flat > y_max)
    out_of_bounds = out_of_bounds_x | out_of_bounds_y

    if extrapolate == "error":
        if torch.any(out_of_bounds):
            raise ExtrapolationError(
                f"Query points outside domain "
                f"[{x_min.item():.4g}, {x_max.item():.4g}] x "
                f"[{y_min.item():.4g}, {y_max.item():.4g}]"
            )
    elif extrapolate == "clamp":
        qx_flat = torch.clamp(qx_flat, x_min, x_max)
        qy_flat = torch.clamp(qy_flat, y_min, y_max)

    # Find intervals
    ix = torch.searchsorted(x_knots[:-1], qx_flat, right=True) - 1
    iy = torch.searchsorted(y_knots[:-1], qy_flat, right=True) - 1

    ix = torch.clamp(ix, 0, x_knots.shape[0] - 2)
    iy = torch.clamp(iy, 0, y_knots.shape[0] - 2)

    # Get local coordinates
    hx = x_knots[1:] - x_knots[:-1]
    hy = y_knots[1:] - y_knots[:-1]

    tx = (qx_flat - x_knots[ix]) / hx[ix]
    ty = (qy_flat - y_knots[iy]) / hy[iy]

    value_shape = coefficients.shape[2:]

    # Get corner values
    z00 = coefficients[ix, iy]
    z10 = coefficients[ix + 1, iy]
    z01 = coefficients[ix, iy + 1]
    z11 = coefficients[ix + 1, iy + 1]

    tx_exp = tx.reshape(-1, *([1] * len(value_shape)))
    ty_exp = ty.reshape(-1, *([1] * len(value_shape)))
    hx_exp = hx[ix].reshape(-1, *([1] * len(value_shape)))
    hy_exp = hy[iy].reshape(-1, *([1] * len(value_shape)))

    if dx == 1 and dy == 0:
        # df/dx using bilinear interpolation derivative
        # f = z00*(1-tx)*(1-ty) + z10*tx*(1-ty) + z01*(1-tx)*ty + z11*tx*ty
        # df/dx = (1/hx) * df/dtx
        # df/dtx = -z00*(1-ty) + z10*(1-ty) - z01*ty + z11*ty
        #        = (z10 - z00)*(1-ty) + (z11 - z01)*ty
        result = ((z10 - z00) * (1 - ty_exp) + (z11 - z01) * ty_exp) / hx_exp

    elif dx == 0 and dy == 1:
        # df/dy
        # df/dty = -z00*(1-tx) - z10*tx + z01*(1-tx) + z11*tx
        #        = (z01 - z00)*(1-tx) + (z11 - z10)*tx
        result = ((z01 - z00) * (1 - tx_exp) + (z11 - z10) * tx_exp) / hy_exp

    elif dx == 1 and dy == 1:
        # d²f/dxdy = (1/hx/hy) * d²f/dtx/dty
        # d²f/dtx/dty = z00 - z10 - z01 + z11
        result = (z00 - z10 - z01 + z11) / (hx_exp * hy_exp)

    elif dx == 2 and dy == 0:
        # For bilinear, second derivative in x is zero
        result = torch.zeros_like(z00)

    elif dx == 0 and dy == 2:
        # For bilinear, second derivative in y is zero
        result = torch.zeros_like(z00)

    else:
        # Higher derivatives are zero for bilinear
        result = torch.zeros_like(z00)

    return result.reshape(*qx.shape, *value_shape)
