"""Hermite spline derivative computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._hermite_spline import HermiteSpline


def hermite_spline_derivative(
    spline: HermiteSpline,
    order: int = 1,
) -> HermiteSpline:
    """
    Compute the derivative of a Hermite spline.

    For the first derivative, this returns a new HermiteSpline where:
    - values become the original derivatives
    - derivatives become the second derivatives (computed from the cubic)

    For higher orders, the process is repeated.

    Parameters
    ----------
    spline : HermiteSpline
        Input Hermite spline
    order : int
        Order of derivative (1, 2, or 3). Default is 1.

    Returns
    -------
    derivative : HermiteSpline
        A new HermiteSpline representing the derivative.

    Raises
    ------
    ValueError
        If order is not 1, 2, or 3.

    Notes
    -----
    For Hermite interpolation on [x_i, x_{i+1}] with h = x_{i+1} - x_i:
    p(x) = H_00(u)*y_i + H_10(u)*h*d_i + H_01(u)*y_{i+1} + H_11(u)*h*d_{i+1}

    where u = (x - x_i) / h.

    The derivative is:
    p'(x) = (1/h) * [H'_00(u)*y_i + H'_10(u)*h*d_i + H'_01(u)*y_{i+1} + H'_11(u)*h*d_{i+1}]

    At u=0 (left endpoint): p'(x_i) = d_i
    At u=1 (right endpoint): p'(x_{i+1}) = d_{i+1}

    For the derivative spline, the new values are the old derivatives,
    and we compute the second derivatives at each knot.
    """
    if order < 1 or order > 3:
        raise ValueError(f"Derivative order must be 1, 2, or 3, got {order}")

    knots = spline.knots
    y_vals = spline.y
    derivatives = spline.dydx

    # Apply derivative transformation 'order' times
    for _ in range(order):
        # New values = old derivatives
        new_values = derivatives.clone()

        # Compute new derivatives (second derivatives) from the cubic
        # At each interior point, we can compute from either the left or right segment
        # For consistency, use the average

        n = knots.shape[0]
        h = knots[1:] - knots[:-1]  # (n-1,)

        # Get value shape
        if y_vals.dim() > 1:
            value_shape = y_vals.shape[1:]
            h_expanded = h.view(-1, *([1] * len(value_shape)))
        else:
            value_shape = ()
            h_expanded = h

        # Second derivative from segment i at u:
        # p''(x) = (1/h^2) * [H''_00(u)*y_i + H''_10(u)*h*d_i + H''_01(u)*y_{i+1} + H''_11(u)*h*d_{i+1}]
        # H''_00(u) = 12u - 6
        # H''_10(u) = 6u - 4
        # H''_01(u) = -12u + 6
        # H''_11(u) = 6u - 2

        # At u=0 (left endpoint of segment):
        # p''(x_i) = (1/h^2) * [-6*y_i - 4*h*d_i + 6*y_{i+1} - 2*h*d_{i+1}]
        #          = (1/h^2) * [6*(y_{i+1} - y_i) - h*(4*d_i + 2*d_{i+1})]
        #          = 6*(y_{i+1} - y_i)/h^2 - (4*d_i + 2*d_{i+1})/h

        # At u=1 (right endpoint of segment):
        # p''(x_{i+1}) = (1/h^2) * [6*y_i + 2*h*d_i - 6*y_{i+1} + 4*h*d_{i+1}]
        #              = (1/h^2) * [-6*(y_{i+1} - y_i) + h*(2*d_i + 4*d_{i+1})]
        #              = -6*(y_{i+1} - y_i)/h^2 + (2*d_i + 4*d_{i+1})/h

        # Compute second derivatives at each knot
        new_derivatives = torch.zeros_like(derivatives)

        # Left endpoint: use first segment
        delta_0 = y_vals[1] - y_vals[0]
        new_derivatives[0] = (
            6 * delta_0 / h_expanded[0] ** 2
            - (4 * derivatives[0] + 2 * derivatives[1]) / h_expanded[0]
        )

        # Interior points: average from left and right segments
        for i in range(1, n - 1):
            # From left segment (at u=1)
            delta_left = y_vals[i] - y_vals[i - 1]
            d2_from_left = (
                -6 * delta_left / h_expanded[i - 1] ** 2
                + (2 * derivatives[i - 1] + 4 * derivatives[i])
                / h_expanded[i - 1]
            )

            # From right segment (at u=0)
            delta_right = y_vals[i + 1] - y_vals[i]
            d2_from_right = (
                6 * delta_right / h_expanded[i] ** 2
                - (4 * derivatives[i] + 2 * derivatives[i + 1]) / h_expanded[i]
            )

            # Average
            new_derivatives[i] = (d2_from_left + d2_from_right) / 2

        # Right endpoint: use last segment
        delta_last = y_vals[-1] - y_vals[-2]
        new_derivatives[-1] = (
            -6 * delta_last / h_expanded[-1] ** 2
            + (2 * derivatives[-2] + 4 * derivatives[-1]) / h_expanded[-1]
        )

        # Update for next iteration
        y_vals = new_values
        derivatives = new_derivatives

    from ._hermite_spline import HermiteSpline

    return HermiteSpline(
        knots=knots.clone(),
        y=y_vals,
        dydx=derivatives,
        extrapolate=spline.extrapolate,
        batch_size=[],
    )


def hermite_spline_derivative_evaluate(
    spline: HermiteSpline,
    t: Tensor,
    order: int = 1,
) -> Tensor:
    """
    Evaluate the derivative of a Hermite spline directly at query points.

    This is more efficient than creating a derivative spline when you
    only need values at specific points.

    Parameters
    ----------
    spline : HermiteSpline
        Input Hermite spline
    t : Tensor
        Query points
    order : int
        Derivative order (1 for first derivative, etc.)

    Returns
    -------
    derivative_values : Tensor
        Derivative values at query points
    """
    if order < 1 or order > 3:
        raise ValueError(f"Derivative order must be 1, 2, or 3, got {order}")

    knots = spline.knots
    y_vals = spline.y
    derivatives = spline.dydx
    extrapolate = spline.extrapolate

    # Check if t is scalar
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    query_shape = t.shape
    t_flat = t.flatten()

    # Get domain bounds
    t_min = knots[0]
    t_max = knots[-1]

    # Handle extrapolation
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside spline domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)

    # Find segment indices
    segment_idx = torch.searchsorted(knots, t_flat, right=True) - 1
    n_segments = len(knots) - 1
    segment_idx = torch.clamp(segment_idx, 0, n_segments - 1)

    # Get interval endpoints and widths
    x_i = knots[segment_idx]
    x_ip1 = knots[segment_idx + 1]
    h = x_ip1 - x_i

    # Compute normalized parameter
    u = (t_flat - x_i) / h

    # Get values and derivatives at segment endpoints
    y_i = y_vals[segment_idx]
    y_ip1 = y_vals[segment_idx + 1]
    d_i = derivatives[segment_idx]
    d_ip1 = derivatives[segment_idx + 1]

    # Get value shape for broadcasting
    if y_vals.dim() > 1:
        value_shape = y_vals.shape[1:]
        u = u.view(-1, *([1] * len(value_shape)))
        h = h.view(-1, *([1] * len(value_shape)))
    else:
        value_shape = ()

    u2 = u * u

    if order == 1:
        # First derivative of Hermite basis functions (scaled by 1/h)
        # H'_00(u) = 6u^2 - 6u
        # H'_10(u) = 3u^2 - 4u + 1
        # H'_01(u) = -6u^2 + 6u
        # H'_11(u) = 3u^2 - 2u
        hp_00 = 6 * u2 - 6 * u
        hp_10 = 3 * u2 - 4 * u + 1
        hp_01 = -6 * u2 + 6 * u
        hp_11 = 3 * u2 - 2 * u

        result = (
            hp_00 * y_i + hp_10 * h * d_i + hp_01 * y_ip1 + hp_11 * h * d_ip1
        ) / h

    elif order == 2:
        # Second derivative of Hermite basis functions (scaled by 1/h^2)
        # H''_00(u) = 12u - 6
        # H''_10(u) = 6u - 4
        # H''_01(u) = -12u + 6
        # H''_11(u) = 6u - 2
        hpp_00 = 12 * u - 6
        hpp_10 = 6 * u - 4
        hpp_01 = -12 * u + 6
        hpp_11 = 6 * u - 2

        result = (
            hpp_00 * y_i
            + hpp_10 * h * d_i
            + hpp_01 * y_ip1
            + hpp_11 * h * d_ip1
        ) / (h**2)

    else:  # order == 3
        # Third derivative is constant on each segment
        # H'''_00(u) = 12
        # H'''_10(u) = 6
        # H'''_01(u) = -12
        # H'''_11(u) = 6
        result = (12 * y_i + 6 * h * d_i - 12 * y_ip1 + 6 * h * d_ip1) / (h**3)

    # Reshape to output shape
    if value_shape:
        result = result.view(*query_shape, *value_shape)
    else:
        result = result.view(*query_shape)

    if is_scalar:
        result = result.squeeze(0)

    return result
