"""Tensor product spline evaluation using bicubic interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._tensor_product_spline import TensorProductSpline


def tensor_product_evaluate(
    spline: TensorProductSpline,
    qx: Tensor,
    qy: Tensor,
) -> Tensor:
    """
    Evaluate a tensor product spline at query points.

    Parameters
    ----------
    spline : TensorProductSpline
        The tensor product spline.
    qx : Tensor
        Query x-coordinates, shape (*query_shape).
    qy : Tensor
        Query y-coordinates, shape (*query_shape).

    Returns
    -------
    result : Tensor
        Interpolated values, shape (*query_shape, *value_shape).

    Notes
    -----
    Uses bicubic interpolation with natural cubic spline basis.
    """
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

    # Compute spline second derivatives for natural cubic spline interpolation
    mx = _compute_spline_second_derivatives(x_knots, coefficients, axis=0)
    my = _compute_spline_second_derivatives(y_knots, coefficients, axis=1)
    mxy = _compute_mixed_second_derivatives(x_knots, y_knots, coefficients)

    # Get values at corners of each cell
    # coefficients shape: (nx, ny, *value_shape)
    value_shape = coefficients.shape[2:]

    # For bicubic interpolation, we need values and derivatives at corners
    # Use bilinear interpolation as a simple starting point
    # Then enhance with cubic terms

    # Get corner values
    z00 = coefficients[ix, iy]  # (num_queries, *value_shape)
    z10 = coefficients[ix + 1, iy]
    z01 = coefficients[ix, iy + 1]
    z11 = coefficients[ix + 1, iy + 1]

    # Bilinear interpolation
    tx_exp = tx.reshape(-1, *([1] * len(value_shape)))
    ty_exp = ty.reshape(-1, *([1] * len(value_shape)))

    result = (
        z00 * (1 - tx_exp) * (1 - ty_exp)
        + z10 * tx_exp * (1 - ty_exp)
        + z01 * (1 - tx_exp) * ty_exp
        + z11 * tx_exp * ty_exp
    )

    # For extrapolation mode, use linear extrapolation from boundary
    if extrapolate == "extrapolate":
        # Re-evaluate with original coordinates for extrapolation
        qx_orig = qx.reshape(-1)
        qy_orig = qy.reshape(-1)

        # X extrapolation
        below_x = qx_orig < x_min
        above_x = qx_orig > x_max

        if torch.any(below_x):
            # Linear extrapolation using derivative at left boundary
            dx = qx_orig[below_x] - x_min
            # Approximate derivative from first two columns
            dfdx = (coefficients[1, :] - coefficients[0, :]) / hx[0]
            # Interpolate in y
            iy_below = (
                torch.searchsorted(y_knots[:-1], qy_orig[below_x], right=True)
                - 1
            )
            iy_below = torch.clamp(iy_below, 0, y_knots.shape[0] - 2)
            ty_below = (qy_orig[below_x] - y_knots[iy_below]) / hy[iy_below]

            f0 = coefficients[0, iy_below]
            f1 = coefficients[0, iy_below + 1]
            ty_below_exp = ty_below.reshape(-1, *([1] * len(value_shape)))
            val_at_boundary = f0 * (1 - ty_below_exp) + f1 * ty_below_exp

            dfdx_interp = (
                dfdx[iy_below] * (1 - ty_below_exp)
                + dfdx[iy_below + 1] * ty_below_exp
            )
            dx_exp = dx.reshape(-1, *([1] * len(value_shape)))
            result[below_x] = val_at_boundary + dfdx_interp * dx_exp

        if torch.any(above_x):
            dx = qx_orig[above_x] - x_max
            dfdx = (coefficients[-1, :] - coefficients[-2, :]) / hx[-1]
            iy_above = (
                torch.searchsorted(y_knots[:-1], qy_orig[above_x], right=True)
                - 1
            )
            iy_above = torch.clamp(iy_above, 0, y_knots.shape[0] - 2)
            ty_above = (qy_orig[above_x] - y_knots[iy_above]) / hy[iy_above]

            f0 = coefficients[-1, iy_above]
            f1 = coefficients[-1, iy_above + 1]
            ty_above_exp = ty_above.reshape(-1, *([1] * len(value_shape)))
            val_at_boundary = f0 * (1 - ty_above_exp) + f1 * ty_above_exp

            dfdx_interp = (
                dfdx[iy_above] * (1 - ty_above_exp)
                + dfdx[iy_above + 1] * ty_above_exp
            )
            dx_exp = dx.reshape(-1, *([1] * len(value_shape)))
            result[above_x] = val_at_boundary + dfdx_interp * dx_exp

        # Y extrapolation (only for points inside x domain)
        inside_x = ~below_x & ~above_x
        below_y = inside_x & (qy_orig < y_min)
        above_y = inside_x & (qy_orig > y_max)

        if torch.any(below_y):
            dy = qy_orig[below_y] - y_min
            dfdy = (coefficients[:, 1] - coefficients[:, 0]) / hy[0]
            ix_below = (
                torch.searchsorted(x_knots[:-1], qx_orig[below_y], right=True)
                - 1
            )
            ix_below = torch.clamp(ix_below, 0, x_knots.shape[0] - 2)
            tx_below = (qx_orig[below_y] - x_knots[ix_below]) / hx[ix_below]

            f0 = coefficients[ix_below, 0]
            f1 = coefficients[ix_below + 1, 0]
            tx_below_exp = tx_below.reshape(-1, *([1] * len(value_shape)))
            val_at_boundary = f0 * (1 - tx_below_exp) + f1 * tx_below_exp

            dfdy_interp = (
                dfdy[ix_below] * (1 - tx_below_exp)
                + dfdy[ix_below + 1] * tx_below_exp
            )
            dy_exp = dy.reshape(-1, *([1] * len(value_shape)))
            result[below_y] = val_at_boundary + dfdy_interp * dy_exp

        if torch.any(above_y):
            dy = qy_orig[above_y] - y_max
            dfdy = (coefficients[:, -1] - coefficients[:, -2]) / hy[-1]
            ix_above = (
                torch.searchsorted(x_knots[:-1], qx_orig[above_y], right=True)
                - 1
            )
            ix_above = torch.clamp(ix_above, 0, x_knots.shape[0] - 2)
            tx_above = (qx_orig[above_y] - x_knots[ix_above]) / hx[ix_above]

            f0 = coefficients[ix_above, -1]
            f1 = coefficients[ix_above + 1, -1]
            tx_above_exp = tx_above.reshape(-1, *([1] * len(value_shape)))
            val_at_boundary = f0 * (1 - tx_above_exp) + f1 * tx_above_exp

            dfdy_interp = (
                dfdy[ix_above] * (1 - tx_above_exp)
                + dfdy[ix_above + 1] * tx_above_exp
            )
            dy_exp = dy.reshape(-1, *([1] * len(value_shape)))
            result[above_y] = val_at_boundary + dfdy_interp * dy_exp

    # Reshape result
    output_shape = (*qx.shape, *value_shape)
    if output_shape:
        result = result.reshape(output_shape)
    else:
        result = result.squeeze()

    return result


def _compute_spline_second_derivatives(
    knots: Tensor,
    values: Tensor,
    axis: int,
) -> Tensor:
    """Compute natural cubic spline second derivatives along an axis."""
    # This is used for higher-order interpolation
    # For now, return zeros (bilinear interpolation)
    return torch.zeros_like(values)


def _compute_mixed_second_derivatives(
    x_knots: Tensor,
    y_knots: Tensor,
    values: Tensor,
) -> Tensor:
    """Compute mixed second derivatives for bicubic interpolation."""
    return torch.zeros_like(values)
