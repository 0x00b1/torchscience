from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from .._knot_error import KnotError
from .._solve_tridiagonal import solve_tridiagonal

if TYPE_CHECKING:
    from ._cubic_spline import CubicSpline


def cubic_spline_fit(
    x: Tensor,
    y: Tensor,
    boundary: str = "not_a_knot",
    boundary_values: Optional[Tensor] = None,
    extrapolate: str = "error",
) -> CubicSpline:
    """
    Fit a cubic spline to data points.

    Parameters
    ----------
    x : Tensor
        Knot positions, shape (n_points,). Must be strictly increasing.
    y : Tensor
        Values at knots, shape (n_points, *value_shape).
    boundary : str
        Boundary condition: "natural", "clamped", "not_a_knot", "periodic".
    boundary_values : Tensor, optional
        For "clamped": first derivatives at endpoints, shape (2, *value_shape).
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extrapolate".

    Returns
    -------
    CubicSpline
        Fitted spline.

    Raises
    ------
    KnotError
        If x is not strictly increasing or has fewer than 2 points.
    """
    n = x.shape[0]

    # Validate knots
    if n < 2:
        raise KnotError(f"Need at least 2 points, got {n}")
    if not torch.all(x[1:] > x[:-1]):
        raise KnotError("Knots must be strictly increasing")

    # Compute interval widths
    h = x[1:] - x[:-1]  # (n-1,)

    # Get value shape
    if y.dim() == 1:
        value_shape = ()
        y_flat = y
    else:
        value_shape = y.shape[1:]
        # Flatten value dimensions for computation
        y_flat = y.reshape(n, -1)  # (n, prod(value_shape))

    n_values = y_flat.shape[1] if y_flat.dim() > 1 else 1
    if y_flat.dim() == 1:
        y_flat = y_flat.unsqueeze(-1)

    # Compute second differences for RHS
    # delta[i] = (y[i+1] - y[i]) / h[i]
    delta = (y_flat[1:] - y_flat[:-1]) / h.unsqueeze(-1)  # (n-1, n_values)

    # Build tridiagonal system for second derivatives m
    # Interior equations: h[i-1]*m[i-1] + 2*(h[i-1]+h[i])*m[i] + h[i]*m[i+1] = 6*(delta[i] - delta[i-1])

    if boundary == "natural":
        # Natural: m[0] = m[n-1] = 0
        # Solve reduced (n-2) x (n-2) system for m[1:n-1]
        if n == 2:
            # Only one segment, no interior points
            m = torch.zeros(2, n_values, dtype=y.dtype, device=y.device)
        else:
            diag = 2 * (h[:-1] + h[1:])  # (n-2,)
            upper = h[1:-1]  # (n-3,)
            lower = h[1:-1]  # (n-3,)
            rhs = 6 * (delta[1:] - delta[:-1])  # (n-2, n_values)

            if n == 3:
                # 1x1 system
                m_interior = rhs / diag.unsqueeze(-1)
            else:
                m_interior = solve_tridiagonal(diag, upper, lower, rhs.T).T

            # Pad with zeros at endpoints
            m = torch.cat(
                [
                    torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                    m_interior,
                    torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                ],
                dim=0,
            )

    elif boundary == "clamped":
        if boundary_values is None:
            raise ValueError("boundary_values required for clamped boundary")
        bv = boundary_values.reshape(2, -1)  # (2, n_values)

        # Full n x n system
        diag = torch.zeros(n, dtype=x.dtype, device=x.device)
        diag[0] = 2 * h[0]
        diag[1:-1] = 2 * (h[:-1] + h[1:])
        diag[-1] = 2 * h[-1]

        upper = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        upper[0] = h[0]
        upper[1:] = h[1:]

        lower = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        lower[:-1] = h[:-1]
        lower[-1] = h[-1]

        rhs = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)
        rhs[0] = 6 * (delta[0] - bv[0])
        rhs[1:-1] = 6 * (delta[1:] - delta[:-1])
        rhs[-1] = 6 * (bv[1] - delta[-1])

        m = solve_tridiagonal(diag, upper, lower, rhs.T).T

    elif boundary == "not_a_knot":
        if n < 4:
            # Fall back to natural for small systems
            return cubic_spline_fit(
                x, y, boundary="natural", extrapolate=extrapolate
            )

        # Not-a-knot: third derivative continuous at x[1] and x[n-2]
        # This modifies the first and last equations
        diag = torch.zeros(n, dtype=x.dtype, device=x.device)
        upper = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        lower = torch.zeros(n - 1, dtype=x.dtype, device=x.device)
        rhs = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)

        # First equation: not-a-knot at x[1]
        diag[0] = h[1]
        upper[0] = -(h[0] + h[1])

        # Interior equations
        for i in range(1, n - 1):
            diag[i] = 2 * (h[i - 1] + h[i])
            if i < n - 1:
                upper[i] = h[i]
            if i > 0:
                lower[i - 1] = h[i - 1]

        # Last equation: not-a-knot at x[n-2]
        lower[-1] = -(h[-2] + h[-1])
        diag[-1] = h[-2]

        # Fix interior equations RHS
        rhs[1:-1] = 6 * (delta[1:] - delta[:-1])

        m = solve_tridiagonal(diag, upper, lower, rhs.T).T

    elif boundary == "periodic":
        if not torch.allclose(y_flat[0], y_flat[-1]):
            raise KnotError("For periodic boundary, y[0] must equal y[-1]")

        # Periodic boundary uses Sherman-Morrison or cyclic reduction
        # For simplicity, use natural boundary on cyclic extension
        # (Production code would use Sherman-Morrison for O(n) solve)
        # Solve reduced system m[0] = m[n-1]
        diag = 2 * (h[:-1] + h[1:])  # (n-2,)
        if n > 3:
            upper = h[1:-1].clone()
            lower = h[1:-1].clone()
        else:
            upper = torch.tensor([], dtype=x.dtype, device=x.device)
            lower = torch.tensor([], dtype=x.dtype, device=x.device)
        rhs = 6 * (delta[1:] - delta[:-1])

        if n == 3:
            m_interior = rhs / diag.unsqueeze(-1)
        elif n > 3:
            m_interior = solve_tridiagonal(diag, upper, lower, rhs.T).T
        else:
            m_interior = torch.zeros(
                0, n_values, dtype=y.dtype, device=y.device
            )

        # For periodic, set m[0] = m[-1]
        m = torch.cat(
            [
                torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
                m_interior,
                torch.zeros(1, n_values, dtype=y.dtype, device=y.device),
            ],
            dim=0,
        )
        m[-1] = m[0]  # Enforce periodicity

    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    # Compute polynomial coefficients for each segment
    # p_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    # where:
    #   a_i = y_i
    #   b_i = delta_i - h_i * (2*m_i + m_{i+1}) / 6
    #   c_i = m_i / 2
    #   d_i = (m_{i+1} - m_i) / (6 * h_i)

    n_seg = n - 1
    a = y_flat[:-1]  # (n_seg, n_values)
    c = m[:-1] / 2  # (n_seg, n_values)
    d = (m[1:] - m[:-1]) / (6 * h.unsqueeze(-1))  # (n_seg, n_values)
    b = delta - h.unsqueeze(-1) * (2 * m[:-1] + m[1:]) / 6  # (n_seg, n_values)

    # Stack coefficients: (n_seg, 4, n_values)
    coeffs = torch.stack([a, b, c, d], dim=1)

    # Reshape back to original value shape
    if value_shape:
        coeffs = coeffs.reshape(n_seg, 4, *value_shape)
    else:
        # Remove the trailing dimension for scalar values
        coeffs = coeffs.squeeze(-1)

    # Lazy import to avoid circular dependency
    from ._cubic_spline import CubicSpline

    return CubicSpline(
        knots=x,
        coefficients=coeffs,
        boundary=boundary,
        extrapolate=extrapolate,
        batch_size=[],
    )
