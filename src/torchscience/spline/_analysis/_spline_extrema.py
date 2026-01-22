"""Spline extrema finding."""

from __future__ import annotations

import torch
from torch import Tensor

from .._cubic_spline import (
    CubicSpline,
    cubic_spline_derivative,
    cubic_spline_evaluate,
)


def spline_extrema(
    spline: CubicSpline,
) -> tuple[Tensor, Tensor]:
    """
    Find local extrema (minima and maxima) of a cubic spline.

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.

    Returns
    -------
    x_extrema : Tensor
        X-coordinates of extrema, sorted.
    y_extrema : Tensor
        Y-values at extrema.

    Notes
    -----
    Finds where the first derivative is zero.
    For multi-valued splines, only works on scalar-valued splines.
    """
    knots = spline.knots
    coefficients = spline.coefficients

    # Only support scalar-valued splines
    if coefficients.dim() > 2:
        raise ValueError("spline_extrema only supports scalar-valued splines")

    n_intervals = knots.shape[0] - 1
    all_extrema = []

    for i in range(n_intervals):
        x0 = knots[i]
        x1 = knots[i + 1]

        # Derivative coefficients: b + 2c*t + 3d*t^2
        b = coefficients[i, 1].item()
        c = coefficients[i, 2].item()
        d = coefficients[i, 3].item()

        h = (x1 - x0).item()

        # Find roots of the derivative (quadratic)
        roots = _quadratic_roots_in_interval(b, 2 * c, 3 * d, h)

        for root in roots:
            global_x = x0.item() + root
            all_extrema.append(global_x)

    if len(all_extrema) == 0:
        empty = torch.tensor([], dtype=knots.dtype, device=knots.device)
        return empty, empty

    x_extrema = torch.tensor(
        all_extrema, dtype=knots.dtype, device=knots.device
    )
    x_extrema = torch.sort(x_extrema)[0]

    # Remove duplicates
    if x_extrema.numel() > 1:
        unique_mask = torch.cat(
            [
                torch.tensor([True], device=x_extrema.device),
                (x_extrema[1:] - x_extrema[:-1]).abs() > 1e-10,
            ]
        )
        x_extrema = x_extrema[unique_mask]

    # Evaluate spline at extrema
    y_extrema = cubic_spline_evaluate(spline, x_extrema)

    return x_extrema, y_extrema


def _quadratic_roots_in_interval(
    a: float, b: float, c: float, h: float
) -> list[float]:
    """
    Find real roots of a + b*t + c*t^2 = 0 in (0, h).

    Note: Excludes endpoints since extrema at knots are handled separately.
    """
    tol = 1e-12
    roots = []

    if abs(c) < tol:
        # Linear
        if abs(b) < tol:
            pass
        else:
            t = -a / b
            if tol < t < h - tol:
                roots.append(t)
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sqrt_disc = disc**0.5
            t1 = (-b - sqrt_disc) / (2 * c)
            t2 = (-b + sqrt_disc) / (2 * c)
            for t in [t1, t2]:
                if tol < t < h - tol:
                    roots.append(t)

    return roots


def spline_minima(spline: CubicSpline) -> tuple[Tensor, Tensor]:
    """
    Find local minima of a cubic spline.

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.

    Returns
    -------
    x_minima : Tensor
        X-coordinates of local minima.
    y_minima : Tensor
        Y-values at local minima.
    """
    x_extrema, y_extrema = spline_extrema(spline)

    if x_extrema.numel() == 0:
        return x_extrema, y_extrema

    # Check second derivative to classify
    deriv2_spline = cubic_spline_derivative(spline, order=2)
    d2y = cubic_spline_evaluate(deriv2_spline, x_extrema)

    # Minimum where d2y > 0
    mask = d2y > 0
    return x_extrema[mask], y_extrema[mask]


def spline_maxima(spline: CubicSpline) -> tuple[Tensor, Tensor]:
    """
    Find local maxima of a cubic spline.

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.

    Returns
    -------
    x_maxima : Tensor
        X-coordinates of local maxima.
    y_maxima : Tensor
        Y-values at local maxima.
    """
    x_extrema, y_extrema = spline_extrema(spline)

    if x_extrema.numel() == 0:
        return x_extrema, y_extrema

    # Check second derivative to classify
    deriv2_spline = cubic_spline_derivative(spline, order=2)
    d2y = cubic_spline_evaluate(deriv2_spline, x_extrema)

    # Maximum where d2y < 0
    mask = d2y < 0
    return x_extrema[mask], y_extrema[mask]
