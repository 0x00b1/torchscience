"""Catmull-Rom spline derivative computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._extrapolation_error import ExtrapolationError

if TYPE_CHECKING:
    from ._catmull_rom import CatmullRomSpline


def catmull_rom_derivative(
    spline: CatmullRomSpline,
    t: Tensor,
    order: int = 1,
) -> Tensor:
    """
    Evaluate the derivative of a Catmull-Rom spline at parameter values.

    Parameters
    ----------
    spline : CatmullRomSpline
        Catmull-Rom spline
    t : Tensor
        Parameter values, shape (*query_shape)
    order : int
        Derivative order (1 or 2). Default is 1.

    Returns
    -------
    derivative_values : Tensor
        Derivative values at parameter values, shape (*query_shape, *value_shape)

    Raises
    ------
    ValueError
        If order is not 1 or 2.

    Notes
    -----
    The derivative is computed by differentiating the Barry-Goldman
    interpolation formula with respect to the parameter.

    For practical purposes, we use numerical differentiation with
    central differences for robustness across all parameterizations.
    """
    if order < 1 or order > 2:
        raise ValueError(f"Derivative order must be 1 or 2, got {order}")

    from ._catmull_rom_evaluate import catmull_rom_evaluate

    control_points = spline.control_points
    extrapolate = spline.extrapolate

    # Check if t is scalar
    is_scalar = t.dim() == 0
    if is_scalar:
        t = t.unsqueeze(0)

    # Store original query shape
    query_shape = t.shape
    t_flat = t.flatten()

    # Get domain bounds
    n_points = control_points.shape[0]
    n_segments = n_points - 3
    t_min = 0.0
    t_max = float(n_segments)

    # Handle extrapolation for boundary checking
    if extrapolate == "error":
        if torch.any(t_flat < t_min) or torch.any(t_flat > t_max):
            raise ExtrapolationError(
                f"Parameter values outside [{t_min}, {t_max}]. "
                "Use extrapolate='clamp' or 'extrapolate'."
            )

    # Use numerical differentiation
    # For first derivative: f'(t) ≈ (f(t+h) - f(t-h)) / (2h)
    # For second derivative: f''(t) ≈ (f(t+h) - 2f(t) + f(t-h)) / h²

    h = 1e-5

    # Create modified spline that allows extrapolation for numerical diff
    from ._catmull_rom import CatmullRomSpline

    spline_for_diff = CatmullRomSpline(
        control_points=control_points,
        alpha=spline.alpha,
        extrapolate="extrapolate",  # Allow extrapolation for numerical diff
        batch_size=[],
    )

    # Clamp evaluation points to a slightly expanded domain for numerical stability
    t_for_eval = t_flat

    if order == 1:
        f_plus = catmull_rom_evaluate(spline_for_diff, t_for_eval + h)
        f_minus = catmull_rom_evaluate(spline_for_diff, t_for_eval - h)
        result = (f_plus - f_minus) / (2 * h)
    else:  # order == 2
        f_plus = catmull_rom_evaluate(spline_for_diff, t_for_eval + h)
        f_center = catmull_rom_evaluate(spline_for_diff, t_for_eval)
        f_minus = catmull_rom_evaluate(spline_for_diff, t_for_eval - h)
        result = (f_plus - 2 * f_center + f_minus) / (h * h)

    # Reshape result
    if control_points.dim() == 1:
        value_shape = ()
    else:
        value_shape = control_points.shape[1:]

    if value_shape:
        result = result.view(*query_shape, *value_shape)
    else:
        result = result.view(*query_shape)

    # Handle scalar input
    if is_scalar:
        result = result.squeeze(0)

    return result
