"""B-spline integration using antiderivative computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._b_spline import BSpline


def b_spline_antiderivative(spline: BSpline) -> BSpline:
    """
    Compute the antiderivative of a B-spline.

    The antiderivative of a B-spline of degree k is a B-spline of degree k+1.
    The new control points are computed using the integral recurrence relation.

    Parameters
    ----------
    spline : BSpline
        Input B-spline of degree k

    Returns
    -------
    antiderivative : BSpline
        A new BSpline of degree k+1 representing the indefinite integral.
        The integration constant is chosen such that the antiderivative
        is 0 at the left boundary of the domain.

    Notes
    -----
    For a B-spline S(t) = Σ c_i B_{i,k}(t), the antiderivative is:

    ∫ S(t) dt = Σ C_i B_{i,k+1}(t)

    The new control points are computed by inverting the derivative formula.
    If derivative gives: d_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
    Then antiderivative gives: C_{i+1} - C_i = c_i * (t_{i+k+1} - t_{i+1}) / (k+1)

    With C_0 = 0 as the integration constant.

    References
    ----------
    .. [1] de Boor, C. "A Practical Guide to Splines", Springer, 2001.
    .. [2] Piegl, L. and Tiller, W. "The NURBS Book", Springer, 1997.
    """
    from ._b_spline import BSpline

    knots = spline.knots
    control_points = spline.control_points
    k = spline.degree
    n_control = control_points.shape[0]

    # The antiderivative has degree k+1
    k_new = k + 1

    # For the antiderivative, we need to extend the knot vector
    # The original spline has n_knots = n_control + k + 1
    # The antiderivative needs n_knots_new = n_control_new + k_new + 1
    # With n_control_new = n_control + 1, we need n_knots_new = n_control + 1 + k + 2 = n_control + k + 3
    # So we need to add 2 knots (one at each end works well for clamped splines)

    # For clamped B-splines, we can extend by repeating the boundary knots
    knots_new = torch.cat(
        [
            knots[:1],  # Repeat first knot
            knots,
            knots[-1:],  # Repeat last knot
        ]
    )

    # Number of new control points
    n_control_new = n_control + 1

    # Handle multi-dimensional control points
    if control_points.dim() == 1:
        value_shape = ()
    else:
        value_shape = control_points.shape[1:]

    # Compute new control points by inverting the derivative formula
    # The derivative formula is: d_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
    # Inverting: c_{i+1} - c_i = d_i * (t_{i+k+1} - t_{i+1}) / k
    #
    # For antiderivative with degree k+1, we have:
    # C_{i+1} - C_i = c_i * (t_{i+k+2} - t_{i+1}) / (k+1)
    #
    # Using knots_new where t'_j = t_{j-1} for j >= 1:
    # For the new spline with n_control_new control points:
    # C_{i+1} - C_i = c_i * (knots_new[i+k+2] - knots_new[i+1]) / (k+1)

    if value_shape:
        control_new = torch.zeros(
            n_control_new, *value_shape, dtype=knots.dtype, device=knots.device
        )
    else:
        control_new = torch.zeros(
            n_control_new, dtype=knots.dtype, device=knots.device
        )

    # C_0 = 0 (integration constant)
    # C_{i+1} = C_i + c_i * (knots_new[i+k+2] - knots_new[i+1]) / (k+1)
    for i in range(n_control):
        scale = (knots_new[i + k_new + 1] - knots_new[i + 1]) / k_new
        control_new[i + 1] = control_new[i] + control_points[i] * scale

    return BSpline(
        knots=knots_new,
        control_points=control_new,
        degree=k_new,
        extrapolate=spline.extrapolate,
        batch_size=[],
    )


def b_spline_integral(
    spline: BSpline,
    a: Union[float, Tensor],
    b: Union[float, Tensor],
) -> Tensor:
    """
    Compute the definite integral of a B-spline from a to b.

    Parameters
    ----------
    spline : BSpline
        Input B-spline
    a : float or Tensor
        Lower bound of integration
    b : float or Tensor
        Upper bound of integration

    Returns
    -------
    integral : Tensor
        Definite integral value(s)

    Notes
    -----
    The integral is computed by:
    1. Computing the antiderivative B-spline F(t)
    2. Evaluating F(b) - F(a)

    The antiderivative is normalized so F(t_min) = 0, where t_min is the
    left boundary of the spline domain.

    Examples
    --------
    >>> import torch
    >>> from torchscience.spline import b_spline_fit, b_spline_integral
    >>> x = torch.linspace(0, 1, 10)
    >>> y = x ** 2  # Quadratic function
    >>> spline = b_spline_fit(x, y, degree=3)
    >>> # Integral of x^2 from 0 to 1 is 1/3
    >>> integral = b_spline_integral(spline, 0.0, 1.0)
    """
    from ._b_spline_evaluate import b_spline_evaluate

    knots = spline.knots
    degree = spline.degree
    control_points = spline.control_points

    # Convert bounds to tensors if necessary
    if not isinstance(a, Tensor):
        a = torch.tensor(a, dtype=knots.dtype, device=knots.device)
    if not isinstance(b, Tensor):
        b = torch.tensor(b, dtype=knots.dtype, device=knots.device)

    # Handle a > b case
    if a > b:
        return -b_spline_integral(spline, b, a)

    # Handle a == b case
    if a == b:
        if control_points.dim() > 1:
            value_shape = control_points.shape[1:]
            return torch.zeros(
                value_shape, dtype=knots.dtype, device=knots.device
            )
        else:
            return torch.tensor(0.0, dtype=knots.dtype, device=knots.device)

    # Compute antiderivative
    antideriv = b_spline_antiderivative(spline)

    # Get domain bounds
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]

    # Clamp integration bounds to domain
    a_clamped = torch.clamp(a, t_min, t_max)
    b_clamped = torch.clamp(b, t_min, t_max)

    # Evaluate antiderivative at bounds
    F_a = b_spline_evaluate(antideriv, a_clamped)
    F_b = b_spline_evaluate(antideriv, b_clamped)

    return F_b - F_a
