"""PCHIP spline evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._pchip import PCHIPSpline


def pchip_evaluate(
    spline: PCHIPSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a PCHIP spline at query points.

    Parameters
    ----------
    spline : PCHIPSpline
        Fitted PCHIP spline from pchip_fit
    t : Tensor
        Query points, shape (*query_shape) or scalar

    Returns
    -------
    y : Tensor
        Interpolated values, shape (*query_shape, *y_dim) where y_dim
        is the dimensionality of the original y values

    Raises
    ------
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'
    """
    knots = spline.knots
    coeffs = spline.coefficients
    extrapolate = spline.extrapolate

    # Check if t is scalar (0-d tensor)
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

    # Find segment indices using searchsorted
    segment_idx = torch.searchsorted(knots, t_flat, right=True) - 1

    # Clamp segment indices to valid range [0, n_segments-1]
    n_segments = len(knots) - 1
    segment_idx = torch.clamp(segment_idx, 0, n_segments - 1)

    # Get the left knot for each query point
    x_i = knots[segment_idx]

    # Compute dx = t - x_i
    dx = t_flat - x_i

    # Get coefficients for each segment
    a = coeffs[segment_idx, 0]
    b = coeffs[segment_idx, 1]
    c = coeffs[segment_idx, 2]
    d = coeffs[segment_idx, 3]

    # Evaluate polynomial using Horner's method: y = a + dx*(b + dx*(c + dx*d))
    if coeffs.dim() > 2:
        value_shape = coeffs.shape[2:]
        dx = dx.view(-1, *([1] * len(value_shape)))
    else:
        value_shape = ()

    y = a + dx * (b + dx * (c + dx * d))

    # Reshape to (*query_shape, *value_shape)
    if value_shape:
        y = y.view(*query_shape, *value_shape)
    else:
        y = y.view(*query_shape)

    # Handle scalar input: return scalar output
    if is_scalar:
        y = y.squeeze(0)

    return y
