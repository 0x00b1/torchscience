"""Smoothing spline evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._smoothing_spline import SmoothingSpline


def smoothing_spline_evaluate(
    spline: SmoothingSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a smoothing spline at query points.

    The smoothing spline is represented by its values at knots.
    Between knots, natural cubic spline interpolation is used.

    Parameters
    ----------
    spline : SmoothingSpline
        Fitted smoothing spline
    t : Tensor
        Query points, shape (*query_shape)

    Returns
    -------
    y : Tensor
        Interpolated values, shape (*query_shape, *value_shape)

    Raises
    ------
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'
    """
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
        coef_2d = coefficients.unsqueeze(-1)  # (n, 1)
    else:
        value_shape = coefficients.shape[1:]
        coef_2d = coefficients.reshape(n, -1)  # (n, num_values)

    # Compute second derivatives for natural cubic spline
    # Natural spline: M_0 = M_{n-1} = 0
    h = knots[1:] - knots[:-1]  # (n-1,)

    # Solve tridiagonal system for second derivatives
    # The system is: h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1} = 6 d_i
    # where d_i = (f_{i+1} - f_i)/h_i - (f_i - f_{i-1})/h_{i-1}

    # Build tridiagonal system for interior M values
    # With natural boundary conditions: M_0 = M_{n-1} = 0

    if n == 2:
        # Linear interpolation
        M = torch.zeros(
            n, coef_2d.shape[1], dtype=knots.dtype, device=knots.device
        )
    else:
        # Compute second differences
        d = torch.zeros(
            n - 2, coef_2d.shape[1], dtype=knots.dtype, device=knots.device
        )
        for i in range(n - 2):
            d[i] = 6.0 * (
                (coef_2d[i + 2] - coef_2d[i + 1]) / h[i + 1]
                - (coef_2d[i + 1] - coef_2d[i]) / h[i]
            )

        # Build and solve tridiagonal system
        # Diagonal: 2(h_{i-1} + h_i)
        # Off-diagonal: h_i
        diag = 2.0 * (h[:-1] + h[1:])  # (n-2,)
        off_diag = h[1:-1]  # (n-3,)

        # Thomas algorithm for tridiagonal system
        M_interior = _solve_tridiagonal(off_diag, diag, off_diag, d)

        # Full M with boundary conditions
        M = torch.zeros(
            n, coef_2d.shape[1], dtype=knots.dtype, device=knots.device
        )
        M[1:-1] = M_interior

    # Find segment indices for each query point
    segment_idx = torch.searchsorted(knots, t_flat, right=True) - 1
    segment_idx = torch.clamp(segment_idx, 0, n - 2)

    # Get segment endpoints
    x_i = knots[segment_idx]
    x_ip1 = knots[segment_idx + 1]
    h_seg = x_ip1 - x_i

    # Get coefficients and second derivatives
    f_i = coef_2d[segment_idx]
    f_ip1 = coef_2d[segment_idx + 1]
    M_i = M[segment_idx]
    M_ip1 = M[segment_idx + 1]

    # Expand for broadcasting
    h_seg = h_seg.unsqueeze(-1)

    # Cubic spline formula:
    # S(x) = M_i (x_{i+1} - x)^3 / (6h_i) + M_{i+1} (x - x_i)^3 / (6h_i)
    #      + (f_i - M_i h_i^2/6) (x_{i+1} - x) / h_i
    #      + (f_{i+1} - M_{i+1} h_i^2/6) (x - x_i) / h_i

    dx_right = (x_ip1 - t_flat).unsqueeze(-1)  # x_{i+1} - x
    dx_left = (t_flat - x_i).unsqueeze(-1)  # x - x_i

    result = (
        M_i * dx_right**3 / (6 * h_seg)
        + M_ip1 * dx_left**3 / (6 * h_seg)
        + (f_i - M_i * h_seg**2 / 6) * dx_right / h_seg
        + (f_ip1 - M_ip1 * h_seg**2 / 6) * dx_left / h_seg
    )

    # Handle linear extrapolation if needed
    if extrapolate == "extrapolate":
        # Linear extrapolation using first derivative at boundaries
        outside_left = t_flat < t_min
        outside_right = t_flat > t_max

        if outside_left.any():
            # First derivative at left boundary
            deriv_0 = (coef_2d[1] - coef_2d[0]) / h[0] - h[0] * (
                2 * M[0] + M[1]
            ) / 6
            delta = (t_flat[outside_left] - t_min).unsqueeze(-1)
            result[outside_left] = coef_2d[0] + deriv_0 * delta

        if outside_right.any():
            # First derivative at right boundary
            deriv_n = (coef_2d[-1] - coef_2d[-2]) / h[-1] + h[-1] * (
                M[-2] + 2 * M[-1]
            ) / 6
            delta = (t_flat[outside_right] - t_max).unsqueeze(-1)
            result[outside_right] = coef_2d[-1] + deriv_n * delta

    # Reshape result
    if value_shape:
        result = result.view(*query_shape, *value_shape)
    else:
        result = result.squeeze(-1).view(*query_shape)

    # Handle scalar input
    if is_scalar:
        result = result.squeeze(0)

    return result


def _solve_tridiagonal(
    lower: Tensor, diag: Tensor, upper: Tensor, rhs: Tensor
) -> Tensor:
    """Solve tridiagonal system using Thomas algorithm.

    Parameters
    ----------
    lower : Tensor
        Lower diagonal, shape (n-1,)
    diag : Tensor
        Main diagonal, shape (n,)
    upper : Tensor
        Upper diagonal, shape (n-1,)
    rhs : Tensor
        Right-hand side, shape (n, m)

    Returns
    -------
    x : Tensor
        Solution, shape (n, m)
    """
    n = diag.shape[0]
    m = rhs.shape[1]

    # Forward elimination
    c_prime = torch.zeros(n - 1, dtype=diag.dtype, device=diag.device)
    d_prime = torch.zeros(n, m, dtype=diag.dtype, device=diag.device)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    d_prime[-1] = (rhs[-1] - lower[-1] * d_prime[-2]) / (
        diag[-1] - lower[-1] * c_prime[-1]
    )

    # Back substitution
    x = torch.zeros(n, m, dtype=diag.dtype, device=diag.device)
    x[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x
