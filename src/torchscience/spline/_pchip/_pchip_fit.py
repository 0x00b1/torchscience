"""PCHIP fitting using the Fritsch-Carlson algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._knot_error import KnotError

if TYPE_CHECKING:
    from ._pchip import PCHIPSpline


def pchip_fit(
    x: Tensor,
    y: Tensor,
    extrapolate: str = "error",
) -> PCHIPSpline:
    """
    Fit a PCHIP spline to data points using the Fritsch-Carlson algorithm.

    Parameters
    ----------
    x : Tensor
        Knot positions, shape (n_points,). Must be strictly increasing.
    y : Tensor
        Values at knots, shape (n_points, *value_shape).
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extend".

    Returns
    -------
    PCHIPSpline
        Fitted spline.

    Raises
    ------
    KnotError
        If x is not strictly increasing or has fewer than 2 points.

    Notes
    -----
    The Fritsch-Carlson algorithm computes slopes that preserve monotonicity:

    1. Compute secants: delta[i] = (y[i+1] - y[i]) / h[i]
    2. Initialize endpoint slopes using one-sided differences
    3. Interior slopes: weighted harmonic mean where signs match, else 0
    4. Limit slopes to ensure alpha^2 + beta^2 <= 9 for monotonicity

    References
    ----------
    Fritsch, F. N. and Carlson, R. E. (1980). "Monotone Piecewise Cubic
    Interpolation". SIAM Journal on Numerical Analysis. 17 (2): 238-246.
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
        y_flat = y.reshape(n, -1)  # (n, prod(value_shape))

    n_values = y_flat.shape[1] if y_flat.dim() > 1 else 1
    if y_flat.dim() == 1:
        y_flat = y_flat.unsqueeze(-1)

    # Compute secants (slopes between consecutive points)
    delta = (y_flat[1:] - y_flat[:-1]) / h.unsqueeze(-1)  # (n-1, n_values)

    # Initialize slopes array
    d = torch.zeros(n, n_values, dtype=y.dtype, device=y.device)

    if n == 2:
        # Only two points: use secant as slope at both ends
        d[0] = delta[0]
        d[1] = delta[0]
    else:
        # Endpoint slopes using one-sided differences
        # Left endpoint: use shape-preserving three-point formula
        d[0] = _edge_slope(h[0], h[1], delta[0], delta[1])

        # Right endpoint
        d[-1] = _edge_slope(h[-1], h[-2], delta[-1], delta[-2])

        # Interior slopes using Fritsch-Carlson
        for i in range(1, n - 1):
            d1 = delta[i - 1]  # left secant
            d2 = delta[i]  # right secant

            # Check sign agreement
            sign_agree = (d1 * d2) > 0

            # Where signs agree, use weighted harmonic mean
            # w1 = 2*h[i] + h[i-1], w2 = h[i] + 2*h[i-1]
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]

            # Weighted harmonic mean: 1/d = w1/(w1+w2)/d1 + w2/(w1+w2)/d2
            # Equivalent: d = (w1 + w2) / (w1/d1 + w2/d2)
            # But we need to handle zeros carefully
            wsum = w1 + w2

            # Compute harmonic mean where signs agree
            with torch.no_grad():
                # Avoid division by zero
                d1_safe = torch.where(d1 == 0, torch.ones_like(d1), d1)
                d2_safe = torch.where(d2 == 0, torch.ones_like(d2), d2)

            harmonic = wsum / (w1 / d1_safe + w2 / d2_safe)

            # Set slope to 0 where signs don't agree or either secant is 0
            d[i] = torch.where(
                sign_agree, harmonic, torch.zeros_like(harmonic)
            )
            d[i] = torch.where(d1 == 0, torch.zeros_like(d[i]), d[i])
            d[i] = torch.where(d2 == 0, torch.zeros_like(d[i]), d[i])

        # Limit slopes to preserve monotonicity (Fritsch-Carlson condition)
        d = _limit_slopes(d, delta, h)

    # Convert slopes to polynomial coefficients
    # p_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3
    # Using Hermite basis conversion:
    #   a = y[i]
    #   b = d[i]  (slope at left)
    #   c = (3*delta - 2*d[i] - d[i+1]) / h
    #   d = (d[i] + d[i+1] - 2*delta) / h^2

    n_seg = n - 1
    a = y_flat[:-1]  # (n_seg, n_values)
    b = d[:-1]  # (n_seg, n_values)

    # c = (3*delta - 2*d[:-1] - d[1:]) / h
    c = (3 * delta - 2 * d[:-1] - d[1:]) / h.unsqueeze(-1)

    # d_coeff = (d[:-1] + d[1:] - 2*delta) / h^2
    d_coeff = (d[:-1] + d[1:] - 2 * delta) / (h.unsqueeze(-1) ** 2)

    # Stack coefficients: (n_seg, 4, n_values)
    coeffs = torch.stack([a, b, c, d_coeff], dim=1)

    # Reshape back to original value shape
    if value_shape:
        coeffs = coeffs.reshape(n_seg, 4, *value_shape)
    else:
        coeffs = coeffs.squeeze(-1)

    from ._pchip import PCHIPSpline

    return PCHIPSpline(
        knots=x,
        coefficients=coeffs,
        extrapolate=extrapolate,
        batch_size=[],
    )


def _edge_slope(
    h1: Tensor, h2: Tensor, delta1: Tensor, delta2: Tensor
) -> Tensor:
    """
    Compute shape-preserving slope at an edge point.

    Uses a three-point formula that preserves monotonicity.

    Parameters
    ----------
    h1 : Tensor
        Width of adjacent interval
    h2 : Tensor
        Width of next interval
    delta1 : Tensor
        Secant of adjacent interval
    delta2 : Tensor
        Secant of next interval

    Returns
    -------
    d : Tensor
        Edge slope
    """
    # Three-point formula
    d = ((2 * h1 + h2) * delta1 - h1 * delta2) / (h1 + h2)

    # Ensure monotonicity: if d and delta1 have different signs, set d = 0
    sign_differ = (d * delta1) < 0
    d = torch.where(sign_differ, torch.zeros_like(d), d)

    # If d and delta1 have same sign but |d| > 3|delta1|, limit d
    sign_same_but_large = (d * delta1 > 0) & (
        torch.abs(d) > 3 * torch.abs(delta1)
    )
    d = torch.where(sign_same_but_large, 3 * delta1, d)

    return d


def _limit_slopes(d: Tensor, delta: Tensor, h: Tensor) -> Tensor:
    """
    Limit slopes to ensure monotonicity (Fritsch-Carlson condition).

    The condition alpha^2 + beta^2 <= 9 must hold where:
    alpha = d[i] / delta[i]
    beta = d[i+1] / delta[i]

    Parameters
    ----------
    d : Tensor
        Slopes at knots, shape (n, n_values)
    delta : Tensor
        Secants, shape (n-1, n_values)
    h : Tensor
        Interval widths, shape (n-1,)

    Returns
    -------
    d : Tensor
        Limited slopes
    """
    n = d.shape[0]
    d = d.clone()

    for i in range(n - 1):
        # Skip if secant is zero (flat region)
        nonzero = delta[i] != 0

        if not nonzero.any():
            continue

        # Compute alpha and beta where secant is nonzero
        alpha = torch.where(nonzero, d[i] / delta[i], torch.zeros_like(d[i]))
        beta = torch.where(
            nonzero, d[i + 1] / delta[i], torch.zeros_like(d[i + 1])
        )

        # Check if alpha^2 + beta^2 > 9
        tau = alpha**2 + beta**2
        needs_limiting = (tau > 9) & nonzero

        if needs_limiting.any():
            # Scale down to satisfy constraint
            scale = 3.0 / torch.sqrt(tau)
            d[i] = torch.where(needs_limiting, scale * alpha * delta[i], d[i])
            d[i + 1] = torch.where(
                needs_limiting, scale * beta * delta[i], d[i + 1]
            )

    return d
