from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from .._degree_error import DegreeError
from .._knot_error import KnotError
from ._b_spline_basis import b_spline_basis

if TYPE_CHECKING:
    from ._b_spline import BSpline


def b_spline_fit(
    x: Tensor,
    y: Tensor,
    degree: int = 3,
    n_knots: Optional[int] = None,
    knots: Optional[Tensor] = None,
    extrapolate: str = "error",
) -> BSpline:
    """
    Fit a B-spline to data points using least squares.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates, shape (n_points,)
    y : Tensor
        Data y-values, shape (n_points,) or (n_points, *y_dim)
    degree : int
        Polynomial degree (default 3 for cubic)
    n_knots : int, optional
        Number of interior knots (uniformly spaced). Mutually exclusive with knots.
    knots : Tensor, optional
        Explicit knot vector. If not provided, creates clamped uniform knots.
    extrapolate : str
        Extrapolation mode for returned BSpline

    Returns
    -------
    spline : BSpline
        Fitted B-spline

    Raises
    ------
    ValueError
        If both n_knots and knots are provided, or if inputs are invalid.
    KnotError
        If knots are not non-decreasing.
    DegreeError
        If degree is negative or too high for the given knot count.

    Notes
    -----
    Solves the least squares problem: min ||B @ c - y||^2
    where B is the collocation matrix (basis functions at data points).
    """
    n_points = x.shape[0]

    # Validate inputs
    if n_points < 2:
        raise ValueError(f"Need at least 2 data points, got {n_points}")

    if degree < 0:
        raise DegreeError(f"Degree must be non-negative, got {degree}")

    if n_knots is not None and knots is not None:
        raise ValueError("Cannot specify both n_knots and knots")

    # Generate knots if not provided
    if knots is None:
        # Determine number of interior knots
        if n_knots is None:
            # Default: min(n_points // 2, 10)
            n_knots = min(n_points // 2, 10)

        # Create clamped uniform knot vector
        # Total knots = n_interior + 2*(degree+1)
        # For n_knots interior knots in [x_min, x_max]:
        # - (degree+1) copies of x_min at start
        # - n_knots uniformly spaced interior knots
        # - (degree+1) copies of x_max at end
        x_min = x.min()
        x_max = x.max()

        if n_knots > 0:
            # Interior knots uniformly spaced between x_min and x_max
            interior_knots = torch.linspace(
                x_min, x_max, n_knots + 2, dtype=x.dtype, device=x.device
            )[1:-1]
        else:
            interior_knots = torch.tensor([], dtype=x.dtype, device=x.device)

        # Build clamped knot vector
        boundary_left = x_min.expand(degree + 1)
        boundary_right = x_max.expand(degree + 1)
        knots = torch.cat([boundary_left, interior_knots, boundary_right])

    # Validate knots
    if not torch.all(knots[1:] >= knots[:-1]):
        raise KnotError("Knots must be non-decreasing")

    n_knots_total = knots.shape[0]
    if n_knots_total < degree + 2:
        raise DegreeError(
            f"Need at least {degree + 2} knots for degree {degree}, got {n_knots_total}"
        )

    # Number of control points = n_knots - degree - 1
    n_control = n_knots_total - degree - 1

    if n_points < n_control:
        raise ValueError(
            f"Need at least {n_control} data points for {n_control} control points, got {n_points}"
        )

    # Build collocation matrix B: B[i, j] = B_j(x[i])
    # Shape: (n_points, n_control)
    B = b_spline_basis(x, knots, degree)

    # Handle y dimensions
    if y.dim() == 1:
        y_flat = y.unsqueeze(-1)  # (n_points, 1)
        is_1d = True
    else:
        y_flat = y.reshape(n_points, -1)  # (n_points, n_values)
        is_1d = False

    n_values = y_flat.shape[1]

    # Solve least squares: min ||B @ c - y||^2
    # Using torch.linalg.lstsq: returns c such that ||B @ c - y||^2 is minimized
    solution = torch.linalg.lstsq(B, y_flat)
    control_points = solution.solution  # (n_control, n_values)

    # Reshape control points back to original y shape
    if is_1d:
        control_points = control_points.squeeze(-1)  # (n_control,)
    else:
        y_dim = y.shape[1:]
        control_points = control_points.reshape(n_control, *y_dim)

    # Lazy import to avoid circular dependency
    from ._b_spline import BSpline

    return BSpline(
        knots=knots,
        control_points=control_points,
        degree=degree,
        extrapolate=extrapolate,
        batch_size=[],
    )
