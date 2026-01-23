"""Bezier curve subdivision (splitting) using De Casteljau's algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._bezier import BezierCurve


def bezier_split(
    curve: BezierCurve,
    t: Union[float, Tensor] = 0.5,
) -> tuple[BezierCurve, BezierCurve]:
    """
    Split a Bezier curve at parameter t into two curves.

    Uses De Casteljau's algorithm to find the control points of the
    two subcurves. The first curve covers [0, t] and the second covers [t, 1].

    Parameters
    ----------
    curve : BezierCurve
        Input Bezier curve
    t : float or Tensor
        Split parameter in [0, 1]. Default is 0.5 (midpoint split).

    Returns
    -------
    left : BezierCurve
        Left subcurve covering the parameter range [0, t]
    right : BezierCurve
        Right subcurve covering the parameter range [t, 1]

    Notes
    -----
    De Casteljau's algorithm naturally produces the control points for
    both subcurves as a byproduct:

    - Left subcurve control points: b_0^(0), b_0^(1), ..., b_0^(n)
      (the leftmost point at each level)
    - Right subcurve control points: b_0^(n), b_1^(n-1), ..., b_n^(0)
      (the diagonal from bottom-left to top-right)

    Examples
    --------
    >>> import torch
    >>> from torchscience.spline import bezier, bezier_split, bezier_evaluate
    >>> control_points = torch.tensor([[0., 0.], [0.5, 1.], [1., 0.]])
    >>> curve = BezierCurve(control_points=control_points, extrapolate="error", batch_size=[])
    >>> left, right = bezier_split(curve, 0.5)
    >>> # left covers [0, 0.5], right covers [0.5, 1]
    >>> # Evaluating at endpoints should match
    >>> bezier_evaluate(left, torch.tensor(1.0))  # Same as original at t=0.5
    >>> bezier_evaluate(right, torch.tensor(0.0))  # Also same as original at t=0.5
    """
    from ._bezier import BezierCurve

    control_points = curve.control_points

    # Convert t to tensor if necessary
    if not isinstance(t, Tensor):
        t = torch.tensor(
            t, dtype=control_points.dtype, device=control_points.device
        )

    n = control_points.shape[0]  # n+1 control points for degree n curve

    if control_points.dim() == 1:
        value_shape = ()
    else:
        value_shape = control_points.shape[1:]

    # De Casteljau's algorithm, storing all intermediate results
    # pyramid[r][i] = b_i^(r)
    pyramid = [control_points.clone()]

    for r in range(1, n):
        prev = pyramid[-1]
        # b_i^(r) = (1-t) * b_i^(r-1) + t * b_{i+1}^(r-1)
        if value_shape:
            t_exp = t.view(*([1] * len(value_shape)))
        else:
            t_exp = t
        new_level = (1 - t_exp) * prev[:-1] + t_exp * prev[1:]
        pyramid.append(new_level)

    # Left subcurve: leftmost points at each level
    # b_0^(0), b_0^(1), ..., b_0^(n-1)
    left_control = torch.stack([level[0] for level in pyramid], dim=0)

    # Right subcurve: rightmost points at each level (reversed)
    # b_0^(n-1), b_1^(n-2), ..., b_{n-1}^(0)
    right_control = torch.stack(
        [level[-1] for level in reversed(pyramid)], dim=0
    )

    left = BezierCurve(
        control_points=left_control,
        extrapolate=curve.extrapolate,
        batch_size=[],
    )

    right = BezierCurve(
        control_points=right_control,
        extrapolate=curve.extrapolate,
        batch_size=[],
    )

    return left, right
