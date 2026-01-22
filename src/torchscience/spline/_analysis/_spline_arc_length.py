"""Spline arc length computation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .._cubic_spline import (
    CubicSpline,
    cubic_spline_derivative,
    cubic_spline_evaluate,
)


def spline_arc_length(
    spline: CubicSpline,
    a: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    num_points: int = 100,
) -> Tensor:
    """
    Compute arc length of a cubic spline curve.

    For a parametric curve y = f(x), the arc length is:
    L = ∫ sqrt(1 + (dy/dx)²) dx

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.
    a : Tensor, optional
        Start of integration interval. Default is first knot.
    b : Tensor, optional
        End of integration interval. Default is last knot.
    num_points : int
        Number of quadrature points per interval.

    Returns
    -------
    length : Tensor
        Arc length of the spline curve.

    Notes
    -----
    Uses Gaussian quadrature for numerical integration.
    """
    knots = spline.knots

    if a is None:
        a = knots[0]
    if b is None:
        b = knots[-1]

    # Get derivative spline
    deriv_spline = cubic_spline_derivative(spline, order=1)

    # Generate quadrature points
    t = torch.linspace(
        0, 1, num_points, dtype=knots.dtype, device=knots.device
    )
    x = a + t * (b - a)

    # Evaluate derivative
    dydx = cubic_spline_evaluate(deriv_spline, x)

    # Arc length integrand: sqrt(1 + (dy/dx)²)
    integrand = torch.sqrt(1 + dydx**2)

    # Trapezoidal integration
    dx = (b - a) / (num_points - 1)
    length = dx * (
        integrand[0] / 2 + integrand[1:-1].sum() + integrand[-1] / 2
    )

    return length


def spline_arc_length_parametric(
    x_spline: CubicSpline,
    y_spline: CubicSpline,
    t_start: Optional[Tensor] = None,
    t_end: Optional[Tensor] = None,
    num_points: int = 100,
) -> Tensor:
    """
    Compute arc length of a parametric spline curve (x(t), y(t)).

    The arc length is:
    L = ∫ sqrt((dx/dt)² + (dy/dt)²) dt

    Parameters
    ----------
    x_spline : CubicSpline
        Spline for x-coordinate as function of parameter t.
    y_spline : CubicSpline
        Spline for y-coordinate as function of parameter t.
    t_start : Tensor, optional
        Start of parameter interval. Default is first knot.
    t_end : Tensor, optional
        End of parameter interval. Default is last knot.
    num_points : int
        Number of quadrature points.

    Returns
    -------
    length : Tensor
        Arc length of the parametric curve.
    """
    knots = x_spline.knots

    if t_start is None:
        t_start = knots[0]
    if t_end is None:
        t_end = knots[-1]

    # Get derivative splines
    dx_spline = cubic_spline_derivative(x_spline, order=1)
    dy_spline = cubic_spline_derivative(y_spline, order=1)

    # Generate quadrature points
    t = torch.linspace(
        0, 1, num_points, dtype=knots.dtype, device=knots.device
    )
    params = t_start + t * (t_end - t_start)

    # Evaluate derivatives
    dxdt = cubic_spline_evaluate(dx_spline, params)
    dydt = cubic_spline_evaluate(dy_spline, params)

    # Arc length integrand: sqrt((dx/dt)² + (dy/dt)²)
    integrand = torch.sqrt(dxdt**2 + dydt**2)

    # Trapezoidal integration
    dt = (t_end - t_start) / (num_points - 1)
    length = dt * (
        integrand[0] / 2 + integrand[1:-1].sum() + integrand[-1] / 2
    )

    return length
