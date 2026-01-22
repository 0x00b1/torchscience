"""Bezier curve derivative computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._bezier import BezierCurve


def bezier_derivative(curve: BezierCurve) -> BezierCurve:
    """
    Compute the derivative of a Bezier curve.

    The derivative of a degree-n Bezier curve is a degree-(n-1) Bezier curve.
    The new control points are:
        Q_i = n * (P_{i+1} - P_i)  for i = 0, ..., n-1

    Parameters
    ----------
    curve : BezierCurve
        Input Bezier curve of degree n >= 1

    Returns
    -------
    derivative : BezierCurve
        Derivative Bezier curve of degree n-1

    Raises
    ------
    ValueError
        If the curve has degree 0 (constant).

    Notes
    -----
    The derivative of a Bezier curve B(t) = Σ P_i B_{i,n}(t) is:

    B'(t) = n * Σ (P_{i+1} - P_i) B_{i,n-1}(t)

    where B_{i,n}(t) are Bernstein basis polynomials.

    This means the derivative is also a Bezier curve, with:
    - Degree reduced by 1
    - Control points Q_i = n * (P_{i+1} - P_i)

    Examples
    --------
    >>> import torch
    >>> from torchscience.spline import bezier, bezier_derivative
    >>> # Linear Bezier (degree 1): P(t) = (1-t)*P0 + t*P1
    >>> control_points = torch.tensor([[0., 0.], [1., 2.]])
    >>> curve = BezierCurve(control_points=control_points, extrapolate="error", batch_size=[])
    >>> deriv = bezier_derivative(curve)
    >>> # Derivative is constant: P1 - P0 = [1, 2]
    >>> deriv.control_points
    tensor([[1., 2.]])
    """
    from ._bezier import BezierCurve

    control_points = curve.control_points
    n = control_points.shape[0] - 1  # degree

    if n < 1:
        raise ValueError("Cannot compute derivative of degree-0 Bezier curve")

    # Q_i = n * (P_{i+1} - P_i)
    new_control_points = n * (control_points[1:] - control_points[:-1])

    return BezierCurve(
        control_points=new_control_points,
        extrapolate=curve.extrapolate,
        batch_size=[],
    )


def bezier_derivative_evaluate(
    curve: BezierCurve,
    t: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Evaluate the derivative of a Bezier curve at parameter values.

    This is more efficient than creating a derivative curve when you
    only need values at specific points.

    Parameters
    ----------
    curve : BezierCurve
        Input Bezier curve
    t : Tensor
        Parameter values
    order : int
        Derivative order (default 1)

    Returns
    -------
    derivative_values : Tensor
        Derivative values at parameter values

    Raises
    ------
    ValueError
        If order > degree of the curve.
    """
    from ._bezier_evaluate import bezier_evaluate

    if order < 1:
        raise ValueError(f"Derivative order must be at least 1, got {order}")

    degree = curve.degree

    if order > degree:
        raise ValueError(
            f"Cannot compute order-{order} derivative of degree-{degree} curve"
        )

    # Apply derivative formula recursively
    current_curve = curve
    for _ in range(order):
        current_curve = bezier_derivative(current_curve)

    return bezier_evaluate(current_curve, t)
