"""Hermite spline fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._knot_error import KnotError

if TYPE_CHECKING:
    from ._hermite_spline import HermiteSpline


def hermite_spline_fit(
    x: Tensor,
    y: Tensor,
    dydx: Tensor,
    extrapolate: str = "error",
) -> HermiteSpline:
    """
    Fit a cubic Hermite spline to data points with specified derivatives.

    Parameters
    ----------
    x : Tensor
        Knot positions, shape (n_points,). Must be strictly increasing.
    y : Tensor
        Values at knots, shape (n_points, *value_shape).
    dydx : Tensor
        First derivatives at knots, shape (n_points, *value_shape).
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extend".

    Returns
    -------
    HermiteSpline
        Fitted spline.

    Raises
    ------
    KnotError
        If x is not strictly increasing or has fewer than 2 points.
    ValueError
        If y and dydx shapes don't match.
    """
    n = x.shape[0]

    # Validate knots
    if n < 2:
        raise KnotError(f"Need at least 2 points, got {n}")
    if not torch.all(x[1:] > x[:-1]):
        raise KnotError("Knots must be strictly increasing")

    # Validate shapes
    if y.shape != dydx.shape:
        raise ValueError(
            f"y and dydx must have same shape, got {y.shape} and {dydx.shape}"
        )

    from ._hermite_spline import HermiteSpline

    return HermiteSpline(
        knots=x.clone(),
        y=y.clone(),
        dydx=dydx.clone(),
        extrapolate=extrapolate,
        batch_size=[],
    )
