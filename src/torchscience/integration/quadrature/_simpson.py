"""Simpson's rule for numerical integration."""

from typing import Optional

import torch
from torch import Tensor


def simpson(
    y: Tensor,
    x: Optional[Tensor] = None,
    *,
    dx: float = 1.0,
    dim: int = -1,
    even: str = "avg",
) -> Tensor:
    """
    Integrate y using composite Simpson's rule.

    Parameters
    ----------
    y : Tensor
        Values to integrate. Must have at least 3 points.
    x : Tensor, optional
        Sample points.
    dx : float
        Spacing when x is None.
    dim : int
        Dimension along which to integrate.
    even : str
        How to handle even number of intervals:
        - "avg": Average of first and last Simpson's rules
        - "first": Use Simpson's rule for first N-2, trapezoid for last
        - "last": Use trapezoid for first, Simpson's rule for last N-2

    Returns
    -------
    Tensor
        Definite integral approximation.

    Raises
    ------
    ValueError
        If y has fewer than 3 points along ``dim``.

    Notes
    -----
    Simpson's rule has O(h^4) error vs O(h^2) for trapezoid.
    Fully differentiable with respect to both ``y`` and ``x``.

    Examples
    --------
    >>> y = torch.sin(torch.linspace(0, torch.pi, 101))
    >>> simpson(y, dx=torch.pi / 100)  # approximately 2.0
    """
    if even not in ("avg", "first", "last"):
        raise ValueError(
            f"even must be 'avg', 'first', or 'last', got '{even}'"
        )

    # Move target dim to end
    y = torch.movedim(y, dim, -1)
    n = y.shape[-1]

    if n < 3:
        raise ValueError(f"simpson requires at least 3 points, got {n}")

    if x is not None:
        x = torch.movedim(x, dim, -1)

    # Number of intervals
    n_intervals = n - 1

    if n_intervals % 2 == 0:
        # Odd number of points (even intervals) - use standard Simpson's 1/3
        result = _simpson_uniform(y, x, dx)
    else:
        # Even number of points (odd intervals) - need special handling
        result = _simpson_handle_even(y, x, dx, even)

    return result


def _simpson_uniform(y: Tensor, x: Optional[Tensor], dx: float) -> Tensor:
    """Simpson's 1/3 rule for odd number of points (even intervals)."""
    if x is not None:
        # Non-uniform spacing: use weighted Simpson's rule
        h = x[..., 1:] - x[..., :-1]
        n = y.shape[-1]

        result = torch.zeros(y.shape[:-1], dtype=y.dtype, device=y.device)

        for i in range(0, n - 2, 2):
            h0 = h[..., i]
            h1 = h[..., i + 1]
            hsum = h0 + h1

            # Simpson's rule for non-uniform mesh
            # Uses Lagrange interpolation weights
            w0 = hsum * (2 * h0 - h1) / (6 * h0)
            w1 = hsum**3 / (6 * h0 * h1)
            w2 = hsum * (2 * h1 - h0) / (6 * h1)

            result = (
                result
                + w0 * y[..., i]
                + w1 * y[..., i + 1]
                + w2 * y[..., i + 2]
            )

        return result
    else:
        # Uniform spacing: standard Simpson's 1/3 rule
        # Integral = (dx/3) * [y0 + 4*y1 + 2*y2 + 4*y3 + ... + 4*y_{n-2} + y_{n-1}]
        return (
            dx
            / 3
            * (
                y[..., 0]
                + 4 * y[..., 1::2].sum(dim=-1)
                + 2 * y[..., 2:-1:2].sum(dim=-1)
                + y[..., -1]
            )
        )


def _simpson_handle_even(
    y: Tensor, x: Optional[Tensor], dx: float, even: str
) -> Tensor:
    """Handle even number of points (odd intervals) using specified strategy."""
    if even == "avg":
        # Average of Simpson from first n-1 points and last n-1 points
        result_first = _simpson_from_slice(y[..., :-1], x, dx, start_idx=0)
        result_last = _simpson_from_slice(y[..., 1:], x, dx, start_idx=1)

        # Add the remaining trapezoid for each
        if x is not None:
            h_last = x[..., -1] - x[..., -2]
            h_first = x[..., 1] - x[..., 0]
        else:
            h_last = dx
            h_first = dx

        trap_last = (y[..., -2] + y[..., -1]) / 2 * h_last
        trap_first = (y[..., 0] + y[..., 1]) / 2 * h_first

        return (result_first + trap_last + result_last + trap_first) / 2

    elif even == "first":
        # Simpson for first n-1 points, trapezoid for last interval
        if x is not None:
            h_last = x[..., -1] - x[..., -2]
        else:
            h_last = dx

        result = _simpson_from_slice(y[..., :-1], x, dx, start_idx=0)
        trap_last = (y[..., -2] + y[..., -1]) / 2 * h_last
        return result + trap_last

    else:  # even == "last"
        # Trapezoid for first interval, Simpson for last n-1 points
        if x is not None:
            h_first = x[..., 1] - x[..., 0]
        else:
            h_first = dx

        trap_first = (y[..., 0] + y[..., 1]) / 2 * h_first
        result = _simpson_from_slice(y[..., 1:], x, dx, start_idx=1)
        return trap_first + result


def _simpson_from_slice(
    y: Tensor, x: Optional[Tensor], dx: float, start_idx: int
) -> Tensor:
    """Apply Simpson's rule to a slice of the data."""
    n = y.shape[-1]
    if n < 3:
        # Fall back to trapezoid for very short arrays
        if x is not None:
            h = x[..., start_idx + n - 1] - x[..., start_idx]
        else:
            h = dx * (n - 1)
        return (y[..., 0] + y[..., -1]) / 2 * h

    if x is not None:
        x_slice = x[..., start_idx : start_idx + n]
        return _simpson_uniform(y, x_slice, dx)
    else:
        return _simpson_uniform(y, None, dx)


def cumulative_simpson(
    y: Tensor,
    x: Optional[Tensor] = None,
    *,
    dx: float = 1.0,
    dim: int = -1,
    initial: Optional[float] = None,
) -> Tensor:
    """
    Cumulatively integrate y using Simpson's rule.

    Uses composite Simpson's rule with running sum.

    Parameters
    ----------
    y : Tensor
        Values to integrate.
    x : Tensor, optional
        Sample points.
    dx : float
        Spacing when x is None.
    dim : int
        Dimension along which to integrate.
    initial : float, optional
        If given, insert this value at the beginning.

    Returns
    -------
    Tensor
        Cumulative integral values.

    Notes
    -----
    For Simpson's rule, we integrate pairs of intervals (3 points at a time).
    The cumulative values are computed at each point.

    Examples
    --------
    >>> y = torch.sin(torch.linspace(0, torch.pi, 101))
    >>> cumulative = cumulative_simpson(y, dx=torch.pi / 100)
    >>> cumulative[-1]  # approximately 2.0
    """
    # Move target dim to end
    y = torch.movedim(y, dim, -1)
    n = y.shape[-1]

    if n < 3:
        # Fall back to trapezoid for very short arrays
        from torchscience.integration.quadrature._trapezoid import (
            cumulative_trapezoid,
        )

        result = cumulative_trapezoid(
            torch.movedim(y, -1, dim), x, dx=dx, dim=dim, initial=initial
        )
        return result

    if x is not None:
        x = torch.movedim(x, dim, -1)

    # Compute Simpson integral for pairs [0,2], [0,4], [0,6], ...
    # Then interpolate for odd indices using trapezoid

    # First, compute cumulative using trapezoid as base
    if x is not None:
        spacing = x[..., 1:] - x[..., :-1]
    else:
        spacing = dx

    avg_y = (y[..., :-1] + y[..., 1:]) / 2
    trap_increments = avg_y * spacing
    trap_cumulative = torch.cumsum(trap_increments, dim=-1)

    # Apply Simpson correction at even indices
    # Simpson is more accurate, so we adjust the cumulative values
    result = trap_cumulative.clone()

    # For uniform spacing, Simpson gives better results
    # We can refine the cumulative by using Simpson's rule where applicable
    if x is None:
        # Uniform spacing: apply Simpson's correction
        # For each pair [2i, 2i+2], Simpson gives:
        # integral = dx/3 * (y[2i] + 4*y[2i+1] + y[2i+2])
        for i in range(2, n, 2):
            # Simpson integral from 0 to i
            simpson_val = _simpson_uniform(y[..., : i + 1], None, dx)
            if i - 1 < result.shape[-1]:
                result[..., i - 1] = simpson_val
    else:
        # Non-uniform: use Simpson where possible
        for i in range(2, n, 2):
            simpson_val = _simpson_uniform(
                y[..., : i + 1], x[..., : i + 1], dx
            )
            if i - 1 < result.shape[-1]:
                result[..., i - 1] = simpson_val

    # Prepend initial value if requested
    if initial is not None:
        initial_tensor = torch.full(
            (*result.shape[:-1], 1),
            initial,
            dtype=result.dtype,
            device=result.device,
        )
        result = torch.cat([initial_tensor, result], dim=-1)

    # Move dim back to original position
    result = torch.movedim(result, -1, dim)

    return result
