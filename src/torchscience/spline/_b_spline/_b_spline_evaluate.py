from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError
from ._b_spline_basis import b_spline_basis

if TYPE_CHECKING:
    from ._b_spline import BSpline


def b_spline_evaluate(
    spline: BSpline,
    t: Tensor,
) -> Tensor:
    """
    Evaluate a B-spline at query points.

    Parameters
    ----------
    spline : BSpline
        B-spline with knots and control points
    t : Tensor
        Query points, shape (*query_shape)

    Returns
    -------
    y : Tensor
        Evaluated values, shape (*query_shape, *y_dim)

    Raises
    ------
    ExtrapolationError
        If any query point is outside the spline domain and
        spline.extrapolate == 'error'

    Notes
    -----
    B-spline evaluation: y(t) = sum_i B_{i,k}(t) * c_i
    where B_{i,k} are basis functions and c_i are control points.
    """
    knots = spline.knots
    control_points = spline.control_points
    degree = spline.degree
    extrapolate = spline.extrapolate

    # Check if t is scalar (0-d tensor)
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Get domain bounds
    # For B-splines, the domain is [knots[degree], knots[n_knots - degree - 1]]
    # For clamped knots (repeated at ends), this is [knots[0], knots[-1]]
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]

    # Handle extrapolation modes
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Query points outside spline domain [{t_min.item()}, {t_max.item()}]"
            )
    elif extrapolate == "clamp":
        t_flat = torch.clamp(t_flat, t_min, t_max)
    # "extrapolate" mode: no special handling, basis functions will handle it

    # Reshape t_flat back to compute basis
    # Compute basis functions: shape (*query_shape, n_control)
    basis = b_spline_basis(t_flat.view(*query_shape), knots, degree)

    # control_points shape: (n_control, *y_dim) or (n_control,) for 1D
    # basis shape: (*query_shape, n_control)
    # Result shape: (*query_shape, *y_dim)

    if control_points.dim() == 1:
        # 1D control points: simple weighted sum
        # basis: (*query_shape, n_control), control_points: (n_control,)
        # einsum: ...i, i -> ...
        y = torch.einsum("...i,i->...", basis, control_points)
    else:
        # Multi-dimensional control points
        # basis: (*query_shape, n_control), control_points: (n_control, *y_dim)
        # We need: y[..., j] = sum_i basis[..., i] * control_points[i, j]
        # This is a matrix multiplication along the control point axis
        y_dim = control_points.shape[1:]
        n_control = control_points.shape[0]

        # Flatten basis to (n_query, n_control) and control_points to (n_control, n_values)
        n_query = basis[..., 0].numel()
        basis_flat = basis.reshape(n_query, n_control)
        control_flat = control_points.reshape(n_control, -1)

        # Matrix multiply: (n_query, n_control) @ (n_control, n_values) -> (n_query, n_values)
        y_flat = torch.mm(basis_flat, control_flat)

        # Reshape back to (*query_shape, *y_dim)
        y = y_flat.view(*query_shape, *y_dim)

    # Handle scalar input: return scalar output (or with y_dim)
    if is_scalar:
        y = y.squeeze(0)

    return y
