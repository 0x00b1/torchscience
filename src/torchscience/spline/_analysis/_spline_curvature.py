"""Spline curvature computation."""

from __future__ import annotations

import torch
from torch import Tensor

from .._cubic_spline import (
    CubicSpline,
    cubic_spline_derivative,
    cubic_spline_evaluate,
)


def spline_curvature(
    spline: CubicSpline,
    t: Tensor,
) -> Tensor:
    """
    Compute curvature of a cubic spline at given points.

    For a curve y = f(x), the curvature is:
    κ = |y''| / (1 + y'²)^(3/2)

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.
    t : Tensor
        Points at which to evaluate curvature.

    Returns
    -------
    curvature : Tensor
        Curvature values at the given points.
        Same shape as t.

    Notes
    -----
    The curvature measures how rapidly the curve deviates from a straight line.
    A straight line has zero curvature; a circle of radius r has curvature 1/r.
    """
    # Get first and second derivative splines
    d1_spline = cubic_spline_derivative(spline, order=1)
    d2_spline = cubic_spline_derivative(spline, order=2)

    # Evaluate derivatives
    dy = cubic_spline_evaluate(d1_spline, t)
    d2y = cubic_spline_evaluate(d2_spline, t)

    # Curvature formula
    curvature = torch.abs(d2y) / (1 + dy**2) ** 1.5

    return curvature


def spline_curvature_parametric(
    x_spline: CubicSpline,
    y_spline: CubicSpline,
    t: Tensor,
) -> Tensor:
    """
    Compute signed curvature of a parametric curve (x(t), y(t)).

    The signed curvature is:
    κ = (x'*y'' - y'*x'') / (x'² + y'²)^(3/2)

    Parameters
    ----------
    x_spline : CubicSpline
        Spline for x-coordinate.
    y_spline : CubicSpline
        Spline for y-coordinate.
    t : Tensor
        Parameter values at which to evaluate curvature.

    Returns
    -------
    curvature : Tensor
        Signed curvature values.
        Positive for counterclockwise, negative for clockwise.
    """
    # Get derivative splines
    dx_spline = cubic_spline_derivative(x_spline, order=1)
    d2x_spline = cubic_spline_derivative(x_spline, order=2)
    dy_spline = cubic_spline_derivative(y_spline, order=1)
    d2y_spline = cubic_spline_derivative(y_spline, order=2)

    # Evaluate derivatives
    dx = cubic_spline_evaluate(dx_spline, t)
    d2x = cubic_spline_evaluate(d2x_spline, t)
    dy = cubic_spline_evaluate(dy_spline, t)
    d2y = cubic_spline_evaluate(d2y_spline, t)

    # Signed curvature formula
    numerator = dx * d2y - dy * d2x
    denominator = (dx**2 + dy**2) ** 1.5

    # Handle near-zero denominator
    curvature = torch.where(
        denominator.abs() > 1e-10,
        numerator / denominator,
        torch.zeros_like(numerator),
    )

    return curvature


def spline_radius_of_curvature(
    spline: CubicSpline,
    t: Tensor,
) -> Tensor:
    """
    Compute radius of curvature of a spline at given points.

    The radius of curvature is the reciprocal of curvature:
    R = 1 / κ = (1 + y'²)^(3/2) / |y''|

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.
    t : Tensor
        Points at which to evaluate radius.

    Returns
    -------
    radius : Tensor
        Radius of curvature values.
        Infinity where curvature is zero (straight line).
    """
    kappa = spline_curvature(spline, t)

    # Radius is reciprocal, handling zero curvature
    radius = torch.where(
        kappa.abs() > 1e-10,
        1.0 / kappa,
        torch.full_like(kappa, float("inf")),
    )

    return radius
