"""Trapezoidal rule for numerical integration."""

from typing import Optional

import torch
from torch import Tensor


def trapezoid(
    y: Tensor,
    x: Optional[Tensor] = None,
    *,
    dx: Optional[float] = None,
    dim: int = -1,
) -> Tensor:
    """
    Integrate y along the given dimension using the composite trapezoidal rule.

    Parameters
    ----------
    y : Tensor
        Values to integrate.
    x : Tensor, optional
        Sample points. If None, uses uniform spacing with ``dx``.
    dx : float, optional
        Spacing between sample points when x is None. Default is 1.0 if neither
        x nor dx is specified.
    dim : int
        Dimension along which to integrate.

    Returns
    -------
    Tensor
        Definite integral approximation. Shape is y.shape with ``dim`` removed.

    Notes
    -----
    Fully differentiable with respect to both ``y`` and ``x``.

    Examples
    --------
    >>> y = torch.sin(torch.linspace(0, torch.pi, 100))
    >>> trapezoid(y, dx=torch.pi / 99)  # approximately 2.0
    """
    if x is not None:
        return torch.trapezoid(y, x, dim=dim)
    elif dx is not None:
        return torch.trapezoid(y, dx=dx, dim=dim)
    else:
        return torch.trapezoid(y, dim=dim)


def cumulative_trapezoid(
    y: Tensor,
    x: Optional[Tensor] = None,
    *,
    dx: float = 1.0,
    dim: int = -1,
    initial: Optional[float] = None,
) -> Tensor:
    """
    Cumulatively integrate y using the composite trapezoidal rule.

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
        If given, insert this value at the beginning. Output has same shape as y.
        If None, output has one fewer element along ``dim``.

    Returns
    -------
    Tensor
        Cumulative integral values.

    Examples
    --------
    >>> y = torch.sin(torch.linspace(0, torch.pi, 100))
    >>> cumulative = cumulative_trapezoid(y, dx=torch.pi / 99)
    >>> cumulative[-1]  # approximately 2.0
    """
    # Move target dim to end for easier indexing
    y = torch.movedim(y, dim, -1)

    if x is not None:
        x = torch.movedim(x, dim, -1)
        spacing = x[..., 1:] - x[..., :-1]
    else:
        spacing = dx

    # Trapezoidal rule: (y[i] + y[i+1]) / 2 * spacing[i]
    avg_y = (y[..., :-1] + y[..., 1:]) / 2
    increments = avg_y * spacing

    # Cumulative sum
    result = torch.cumsum(increments, dim=-1)

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
