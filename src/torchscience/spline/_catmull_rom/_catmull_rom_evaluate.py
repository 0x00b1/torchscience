"""Catmull-Rom spline evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._catmull_rom import CatmullRomSpline


def catmull_rom_evaluate(
    spline: CatmullRomSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a Catmull-Rom spline at parameter values.

    Parameters
    ----------
    spline : CatmullRomSpline
        Catmull-Rom spline with control points
    t : Tensor
        Parameter values, shape (*query_shape).
        Valid range is [0, n-3] for n control points.

    Returns
    -------
    points : Tensor
        Evaluated points, shape (*query_shape, *value_shape)

    Raises
    ------
    ExtrapolationError
        If any parameter is outside valid range and spline.extrapolate == 'error'

    Notes
    -----
    The Catmull-Rom spline with parameterization alpha uses the formula:

    For points P0, P1, P2, P3:
    - Compute knot intervals: t_i = |P_{i+1} - P_i|^alpha
    - Compute tangents using the Barry-Goldman formulation
    - Interpolate using Hermite basis functions

    The centripetal variant (alpha=0.5) is recommended as it avoids
    cusps and self-intersections.

    References
    ----------
    .. [1] Barry, P.J. and Goldman, R.N. "A Recursive Evaluation Algorithm
           for a Class of Catmull-Rom Splines", SIGGRAPH 1988.
    """
    control_points = spline.control_points
    alpha = spline.alpha
    extrapolate = spline.extrapolate

    n_points = control_points.shape[0]
    n_segments = n_points - 3  # Number of curve segments

    if n_points < 4:
        raise ValueError(
            f"Catmull-Rom requires at least 4 control points, got {n_points}"
        )

    # Check if t is scalar
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Valid parameter range: [0, n_segments]
    t_min = 0.0
    t_max = float(n_segments)

    # Handle extrapolation
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Parameter values outside [{t_min}, {t_max}]. "
                "Use extrapolate='clamp' or 'extrapolate'."
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)

    # Get value shape
    if control_points.dim() == 1:
        value_shape = ()
    else:
        value_shape = control_points.shape[1:]

    # Find segment indices
    segment_idx = torch.floor(t_flat).long()
    segment_idx = torch.clamp(segment_idx, 0, n_segments - 1)

    # Local parameter within segment [0, 1]
    u = t_flat - segment_idx.float()

    # Get the 4 control points for each query
    # P0, P1, P2, P3 for segment i are control_points[i:i+4]
    idx0 = segment_idx
    idx1 = segment_idx + 1
    idx2 = segment_idx + 2
    idx3 = segment_idx + 3

    P0 = control_points[idx0]
    P1 = control_points[idx1]
    P2 = control_points[idx2]
    P3 = control_points[idx3]

    # Compute knot intervals based on alpha
    # t_i = |P_{i+1} - P_i|^alpha
    if alpha == 0.0:
        # Uniform: all intervals equal
        t01 = torch.ones_like(t_flat)
        t12 = torch.ones_like(t_flat)
        t23 = torch.ones_like(t_flat)
    else:
        # Non-uniform: interval proportional to distance^alpha
        def compute_interval(Pa, Pb):
            diff = Pb - Pa
            if value_shape:
                dist = torch.norm(diff, dim=-1)
            else:
                dist = diff.abs()
            return dist.pow(alpha)

        t01 = compute_interval(P0, P1)
        t12 = compute_interval(P1, P2)
        t23 = compute_interval(P2, P3)

        # Handle zero intervals (coincident points)
        eps = 1e-8
        t01 = torch.where(t01 < eps, torch.ones_like(t01), t01)
        t12 = torch.where(t12 < eps, torch.ones_like(t12), t12)
        t23 = torch.where(t23 < eps, torch.ones_like(t23), t23)

    # Barry-Goldman formulation for Catmull-Rom
    # Map u in [0, 1] to the knot interval [t1, t2]
    # where t0 = 0, t1 = t01, t2 = t01 + t12, t3 = t01 + t12 + t23

    t0 = torch.zeros_like(t_flat)
    t1 = t01
    t2 = t01 + t12
    t3 = t01 + t12 + t23

    # Parameter in the knot space
    s = t1 + u * t12

    # Expand for broadcasting with value dimensions
    if value_shape:
        expand_dims = [-1] + [1] * len(value_shape)
        t0 = t0.view(*expand_dims)
        t1 = t1.view(*expand_dims)
        t2 = t2.view(*expand_dims)
        t3 = t3.view(*expand_dims)
        s = s.view(*expand_dims)

    # First level of interpolation
    A1 = (t1 - s) / (t1 - t0) * P0 + (s - t0) / (t1 - t0) * P1
    A2 = (t2 - s) / (t2 - t1) * P1 + (s - t1) / (t2 - t1) * P2
    A3 = (t3 - s) / (t3 - t2) * P2 + (s - t2) / (t3 - t2) * P3

    # Second level of interpolation
    B1 = (t2 - s) / (t2 - t0) * A1 + (s - t0) / (t2 - t0) * A2
    B2 = (t3 - s) / (t3 - t1) * A2 + (s - t1) / (t3 - t1) * A3

    # Final interpolation
    C = (t2 - s) / (t2 - t1) * B1 + (s - t1) / (t2 - t1) * B2

    # Reshape result
    if value_shape:
        result = C.view(*query_shape, *value_shape)
    else:
        result = C.view(*query_shape)

    # Handle scalar input
    if is_scalar:
        result = result.squeeze(0)

    return result
