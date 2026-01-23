"""Hermite spline interpolation with user-specified derivatives."""

from typing import Callable

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from ._hermite_spline_evaluate import hermite_spline_evaluate
from ._hermite_spline_fit import hermite_spline_fit


@tensorclass
class HermiteSpline:
    """Cubic Hermite spline with user-specified derivatives.

    Hermite interpolation passes through data points with specified
    first derivatives at each point, providing full control over the
    shape of the curve.

    Attributes
    ----------
    knots : Tensor
        Breakpoints, shape (n_knots,). Strictly increasing.
    y : Tensor
        Values at knots, shape (n_knots, *value_shape).
    dydx : Tensor
        First derivatives at knots, shape (n_knots, *value_shape).
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extend".
    """

    knots: Tensor
    y: Tensor
    dydx: Tensor
    extrapolate: str


def hermite_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    dydx: torch.Tensor,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a cubic Hermite spline interpolator from data and derivatives.

    The Hermite spline passes through all data points with the specified
    first derivatives, giving full control over the shape of the curve.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates. Must be strictly monotonically increasing.
    y : Tensor
        Data y-values. Shape must be compatible with x.
    dydx : Tensor
        First derivatives at each point. Same shape as y.
    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError (default).
        - ``"clamp"``: Clamp to boundary values.
        - ``"extend"``: Extrapolate using boundary polynomial.

    Returns
    -------
    spline : Callable[[Tensor], Tensor]
        Function that evaluates the spline at given points.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 1, 5)
    >>> y = torch.sin(x * 2 * torch.pi)
    >>> dydx = 2 * torch.pi * torch.cos(x * 2 * torch.pi)  # Exact derivative
    >>> f = hermite_spline(x, y, dydx)
    >>> f(torch.tensor([0.5]))  # Evaluate at x=0.5
    """
    fitted = hermite_spline_fit(x, y, dydx, extrapolate=extrapolate)
    return lambda t: hermite_spline_evaluate(fitted, t)
