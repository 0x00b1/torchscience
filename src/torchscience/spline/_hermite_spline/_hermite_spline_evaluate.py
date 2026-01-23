"""Hermite spline evaluation using cubic Hermite basis functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._hermite_spline import HermiteSpline


def hermite_spline_evaluate(
    spline: HermiteSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a Hermite spline at query points using Hermite basis functions.

    The cubic Hermite basis functions for t in [0, 1] are:
    - H_00(t) = (1 + 2t)(1-t)^2  -- value at left endpoint
    - H_10(t) = t(1-t)^2         -- derivative at left endpoint
    - H_01(t) = t^2(3-2t)        -- value at right endpoint
    - H_11(t) = t^2(t-1)         -- derivative at right endpoint

    The interpolant on segment [x_i, x_{i+1}] is:
    p(x) = H_00(u)*y_i + H_10(u)*h*d_i + H_01(u)*y_{i+1} + H_11(u)*h*d_{i+1}

    where u = (x - x_i) / h and h = x_{i+1} - x_i.

    Parameters
    ----------
    spline : HermiteSpline
        Fitted Hermite spline from hermite_spline_fit
    t : Tensor
        Query points, shape (*query_shape) or scalar

    Returns
    -------
    y : Tensor
        Interpolated values, shape (*query_shape, *y_dim)

    Raises
    ------
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'
    """
    knots = spline.knots
    y_vals = spline.y
    derivatives = spline.dydx
    extrapolate = spline.extrapolate

    # Check if t is scalar
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Get domain bounds
    t_min = knots[0]
    t_max = knots[-1]

    # Handle extrapolation modes
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside spline domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)

    # Find segment indices
    segment_idx = torch.searchsorted(knots, t_flat, right=True) - 1

    # Clamp segment indices to valid range
    n_segments = len(knots) - 1
    segment_idx = torch.clamp(segment_idx, 0, n_segments - 1)

    # Get interval endpoints and widths
    x_i = knots[segment_idx]
    x_ip1 = knots[segment_idx + 1]
    h = x_ip1 - x_i

    # Compute normalized parameter u = (t - x_i) / h in [0, 1]
    u = (t_flat - x_i) / h

    # Get values and derivatives at segment endpoints
    y_i = y_vals[segment_idx]
    y_ip1 = y_vals[segment_idx + 1]
    d_i = derivatives[segment_idx]
    d_ip1 = derivatives[segment_idx + 1]

    # Get value shape for broadcasting
    if y_vals.dim() > 1:
        value_shape = y_vals.shape[1:]
        # Expand u and h for broadcasting
        u = u.view(-1, *([1] * len(value_shape)))
        h = h.view(-1, *([1] * len(value_shape)))
    else:
        value_shape = ()

    # Compute Hermite basis functions
    # H_00(u) = (1 + 2u)(1-u)^2 = 2u^3 - 3u^2 + 1
    # H_10(u) = u(1-u)^2 = u^3 - 2u^2 + u
    # H_01(u) = u^2(3-2u) = -2u^3 + 3u^2
    # H_11(u) = u^2(u-1) = u^3 - u^2
    u2 = u * u
    u3 = u2 * u

    h_00 = 2 * u3 - 3 * u2 + 1
    h_10 = u3 - 2 * u2 + u
    h_01 = -2 * u3 + 3 * u2
    h_11 = u3 - u2

    # Evaluate interpolant
    # p(x) = H_00*y_i + H_10*h*d_i + H_01*y_{i+1} + H_11*h*d_{i+1}
    y = h_00 * y_i + h_10 * h * d_i + h_01 * y_ip1 + h_11 * h * d_ip1

    # Reshape to (*query_shape, *value_shape)
    if value_shape:
        y = y.view(*query_shape, *value_shape)
    else:
        y = y.view(*query_shape)

    # Handle scalar input
    if is_scalar:
        y = y.squeeze(0)

    return y
