"""Spline root finding."""

from __future__ import annotations

import torch
from torch import Tensor

from .._cubic_spline import CubicSpline


def spline_roots(
    spline: CubicSpline,
    value: float = 0.0,
) -> Tensor:
    """
    Find roots of a cubic spline (where spline(x) = value).

    Parameters
    ----------
    spline : CubicSpline
        The cubic spline.
    value : float
        Target value to find. Default is 0.

    Returns
    -------
    roots : Tensor
        X-coordinates where spline equals value, sorted in increasing order.
        Shape (num_roots,).

    Notes
    -----
    Uses analytical cubic root finding within each interval.
    For multi-valued splines, only works on scalar-valued splines.
    """
    knots = spline.knots
    coefficients = spline.coefficients

    # Only support scalar-valued splines
    if coefficients.dim() > 2:
        raise ValueError("spline_roots only supports scalar-valued splines")

    n_intervals = knots.shape[0] - 1
    all_roots = []

    for i in range(n_intervals):
        x0 = knots[i]
        x1 = knots[i + 1]

        # Get coefficients for this interval: a + b*t + c*t^2 + d*t^3
        # where t = x - x0
        a = coefficients[i, 0].item() - value
        b = coefficients[i, 1].item()
        c = coefficients[i, 2].item()
        d = coefficients[i, 3].item()

        # Find roots of the cubic in [0, h] where h = x1 - x0
        h = (x1 - x0).item()
        roots_in_interval = _cubic_roots_in_interval(a, b, c, d, h)

        # Convert local coordinates to global
        for root in roots_in_interval:
            global_root = x0.item() + root
            all_roots.append(global_root)

    if len(all_roots) == 0:
        return torch.tensor([], dtype=knots.dtype, device=knots.device)

    roots = torch.tensor(all_roots, dtype=knots.dtype, device=knots.device)
    roots = torch.sort(roots)[0]

    # Remove duplicates (roots at interval boundaries)
    if roots.numel() > 1:
        unique_mask = torch.cat(
            [
                torch.tensor([True], device=roots.device),
                (roots[1:] - roots[:-1]).abs() > 1e-10,
            ]
        )
        roots = roots[unique_mask]

    return roots


def _cubic_roots_in_interval(
    a: float, b: float, c: float, d: float, h: float
) -> list[float]:
    """
    Find real roots of a + b*t + c*t^2 + d*t^3 = 0 in [0, h].

    Parameters
    ----------
    a, b, c, d : float
        Cubic coefficients.
    h : float
        Interval length.

    Returns
    -------
    roots : list of float
        Roots in [0, h].
    """
    tol = 1e-12
    roots = []

    if abs(d) < tol:
        # Quadratic or lower
        if abs(c) < tol:
            # Linear or constant
            if abs(b) < tol:
                # Constant: only root if a ≈ 0
                pass
            else:
                # Linear: a + b*t = 0 => t = -a/b
                t = -a / b
                if 0 <= t <= h:
                    roots.append(t)
        else:
            # Quadratic: a + b*t + c*t^2 = 0
            disc = b * b - 4 * a * c
            if disc >= 0:
                sqrt_disc = disc**0.5
                t1 = (-b - sqrt_disc) / (2 * c)
                t2 = (-b + sqrt_disc) / (2 * c)
                for t in [t1, t2]:
                    if 0 <= t <= h:
                        roots.append(t)
    else:
        # Full cubic - use Cardano's formula or numerical approach
        # Normalize: t^3 + (c/d)*t^2 + (b/d)*t + (a/d) = 0
        # Let t = u - c/(3d) to eliminate quadratic term
        p = (3 * d * b - c * c) / (3 * d * d)
        q = (2 * c * c * c - 9 * d * c * b + 27 * d * d * a) / (27 * d * d * d)

        disc = (q / 2) ** 2 + (p / 3) ** 3

        shift = c / (3 * d)

        if disc > tol:
            # One real root
            sqrt_disc = disc**0.5
            u = _cbrt(-q / 2 + sqrt_disc)
            v = _cbrt(-q / 2 - sqrt_disc)
            t = u + v - shift
            if -tol <= t <= h + tol:
                roots.append(max(0, min(h, t)))
        elif disc < -tol:
            # Three real roots
            r = (-(p**3) / 27) ** 0.5
            theta = _acos(-q / (2 * r)) / 3
            r_cbrt = r ** (1 / 3)

            for k in range(3):
                t = (
                    2 * r_cbrt * _cos(theta + 2 * k * 3.141592653589793 / 3)
                    - shift
                )
                if -tol <= t <= h + tol:
                    roots.append(max(0, min(h, t)))
        else:
            # One or two real roots (multiple root)
            if abs(p) < tol:
                t = -shift
                if -tol <= t <= h + tol:
                    roots.append(max(0, min(h, t)))
            else:
                t1 = 3 * q / p - shift
                t2 = -3 * q / (2 * p) - shift
                for t in [t1, t2]:
                    if -tol <= t <= h + tol:
                        roots.append(max(0, min(h, t)))

    return roots


def _cbrt(x: float) -> float:
    """Cube root that handles negative numbers."""
    if x >= 0:
        return x ** (1 / 3)
    else:
        return -((-x) ** (1 / 3))


def _cos(x: float) -> float:
    """Cosine function."""
    import math

    return math.cos(x)


def _acos(x: float) -> float:
    """Arc cosine, clamped to [-1, 1]."""
    import math

    return math.acos(max(-1, min(1, x)))
