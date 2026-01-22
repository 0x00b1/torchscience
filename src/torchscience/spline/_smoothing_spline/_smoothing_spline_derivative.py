"""Smoothing spline derivative computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._smoothing_spline import SmoothingSpline


def smoothing_spline_derivative(
    spline: SmoothingSpline,
    t: Tensor,
    order: int = 1,
) -> Tensor:
    """
    Evaluate the derivative of a smoothing spline at query points.

    Parameters
    ----------
    spline : SmoothingSpline
        Fitted smoothing spline
    t : Tensor
        Query points, shape (*query_shape)
    order : int
        Derivative order (1 or 2). Default is 1.

    Returns
    -------
    derivative : Tensor
        Derivative values, shape (*query_shape, *value_shape)

    Raises
    ------
    ValueError
        If order is not 1 or 2.
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'
    """
    if order < 1 or order > 2:
        raise ValueError(f"Derivative order must be 1 or 2, got {order}")

    knots = spline.knots
    coefficients = spline.coefficients
    extrapolate = spline.extrapolate

    n = knots.shape[0]

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

    # Handle extrapolation
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside spline domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)

    # Get value shape
    if coefficients.dim() == 1:
        value_shape = ()
        coef_2d = coefficients.unsqueeze(-1)
    else:
        value_shape = coefficients.shape[1:]
        coef_2d = coefficients.reshape(n, -1)

    # Compute second derivatives (same as in evaluate)
    h = knots[1:] - knots[:-1]

    if n == 2:
        M = torch.zeros(
            n, coef_2d.shape[1], dtype=knots.dtype, device=knots.device
        )
    else:
        d = torch.zeros(
            n - 2, coef_2d.shape[1], dtype=knots.dtype, device=knots.device
        )
        for i in range(n - 2):
            d[i] = 6.0 * (
                (coef_2d[i + 2] - coef_2d[i + 1]) / h[i + 1]
                - (coef_2d[i + 1] - coef_2d[i]) / h[i]
            )

        diag = 2.0 * (h[:-1] + h[1:])
        off_diag = h[1:-1]

        from ._smoothing_spline_evaluate import _solve_tridiagonal

        M_interior = _solve_tridiagonal(off_diag, diag, off_diag, d)
        M = torch.zeros(
            n, coef_2d.shape[1], dtype=knots.dtype, device=knots.device
        )
        M[1:-1] = M_interior

    # Find segment indices
    segment_idx = torch.searchsorted(knots, t_flat, right=True) - 1
    segment_idx = torch.clamp(segment_idx, 0, n - 2)

    # Get segment data
    x_i = knots[segment_idx]
    x_ip1 = knots[segment_idx + 1]
    h_seg = x_ip1 - x_i

    f_i = coef_2d[segment_idx]
    f_ip1 = coef_2d[segment_idx + 1]
    M_i = M[segment_idx]
    M_ip1 = M[segment_idx + 1]

    h_seg = h_seg.unsqueeze(-1)
    dx_right = (x_ip1 - t_flat).unsqueeze(-1)
    dx_left = (t_flat - x_i).unsqueeze(-1)

    if order == 1:
        # First derivative of cubic spline:
        # S'(x) = -M_i (x_{i+1} - x)^2 / (2h) + M_{i+1} (x - x_i)^2 / (2h)
        #       - (f_i - M_i h^2/6) / h + (f_{i+1} - M_{i+1} h^2/6) / h
        result = (
            -M_i * dx_right**2 / (2 * h_seg)
            + M_ip1 * dx_left**2 / (2 * h_seg)
            - (f_i - M_i * h_seg**2 / 6) / h_seg
            + (f_ip1 - M_ip1 * h_seg**2 / 6) / h_seg
        )
    else:  # order == 2
        # Second derivative:
        # S''(x) = M_i (x_{i+1} - x) / h + M_{i+1} (x - x_i) / h
        result = M_i * dx_right / h_seg + M_ip1 * dx_left / h_seg

    # Reshape result
    if value_shape:
        result = result.view(*query_shape, *value_shape)
    else:
        result = result.squeeze(-1).view(*query_shape)

    if is_scalar:
        result = result.squeeze(0)

    return result
