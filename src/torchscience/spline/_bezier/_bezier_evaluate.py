"""Bezier curve evaluation using De Casteljau's algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._bezier import BezierCurve


def bezier_evaluate(
    curve: BezierCurve,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a Bezier curve at parameter values using De Casteljau's algorithm.

    De Casteljau's algorithm is numerically stable and works for any degree.
    It recursively computes linear interpolations between adjacent control points.

    Parameters
    ----------
    curve : BezierCurve
        Bezier curve with control points
    t : Tensor
        Parameter values, shape (*query_shape). Should be in [0, 1] for
        standard evaluation.

    Returns
    -------
    points : Tensor
        Evaluated points, shape (*query_shape, *value_shape)

    Raises
    ------
    ExtrapolationError
        If any parameter is outside [0, 1] and curve.extrapolate == 'error'

    Notes
    -----
    De Casteljau's algorithm for control points P_0, P_1, ..., P_n:

    1. Set b_i^(0) = P_i for i = 0, ..., n
    2. For r = 1, ..., n:
       b_i^(r) = (1-t) * b_i^(r-1) + t * b_{i+1}^(r-1)  for i = 0, ..., n-r
    3. Result: B(t) = b_0^(n)

    This is equivalent to evaluating the Bernstein polynomial form but is
    more numerically stable, especially for high-degree curves.
    """
    control_points = curve.control_points
    extrapolate = curve.extrapolate

    # Check if t is scalar
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Handle extrapolation
    if extrapolate == "error":
        if torch.any(t_flat < 0) or torch.any(t_flat > 1):
            raise ExtrapolationError(
                "Parameter values outside [0, 1]. Use extrapolate='clamp' or 'extrapolate'."
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, 0.0, 1.0)
    # "extrapolate" mode: no clamping, allow extrapolation

    # Get dimensions
    n_points = t_flat.shape[0]
    n_control = control_points.shape[0]

    if control_points.dim() == 1:
        value_shape = ()
    else:
        value_shape = control_points.shape[1:]

    # De Casteljau's algorithm
    # Initialize working array with control points
    # Shape: (n_points, n_control, *value_shape)
    if value_shape:
        work = (
            control_points.unsqueeze(0)
            .expand(n_points, -1, *value_shape)
            .clone()
        )
        t_exp = t_flat.view(-1, 1, *([1] * len(value_shape)))
    else:
        work = control_points.unsqueeze(0).expand(n_points, -1).clone()
        t_exp = t_flat.view(-1, 1)

    # Recursive interpolation
    for r in range(1, n_control):
        # Number of points at this level: n_control - r
        # b_i^(r) = (1-t) * b_i^(r-1) + t * b_{i+1}^(r-1)
        work_left = work[:, : n_control - r]
        work_right = work[:, 1 : n_control - r + 1]
        work = (1 - t_exp) * work_left + t_exp * work_right

    # Result is work[:, 0], shape (n_points, *value_shape)
    result = work[:, 0]

    # Reshape to (*query_shape, *value_shape)
    if value_shape:
        result = result.view(*query_shape, *value_shape)
    else:
        result = result.view(*query_shape)

    # Handle scalar input
    if is_scalar:
        result = result.squeeze(0)

    return result
