"""Cubic spline interpolation."""

from typing import Callable

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from ._cubic_spline_evaluate import cubic_spline_evaluate
from ._cubic_spline_fit import cubic_spline_fit


@tensorclass
class CubicSpline:
    """Piecewise cubic polynomial interpolant.

    Attributes
    ----------
    knots : Tensor
        Breakpoints, shape (n_knots,). Strictly increasing.
    coefficients : Tensor
        Polynomial coefficients, shape (n_segments, 4, *value_shape).
        For segment i, the polynomial is:
        a[i] + b[i]*(t-knots[i]) + c[i]*(t-knots[i])^2 + d[i]*(t-knots[i])^3
        where coefficients[i] = [a, b, c, d].
    boundary : str
        Boundary condition type: "natural", "clamped", "not_a_knot", "periodic".
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extrapolate".
    """

    knots: Tensor
    coefficients: Tensor
    boundary: str
    extrapolate: str


def cubic_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    boundary: str = "not_a_knot",
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a cubic spline interpolator from data.

    This is a convenience function that fits a cubic spline and returns
    a callable that evaluates it.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates. Must be strictly monotonically increasing.
    y : Tensor
        Data y-values. Shape must be compatible with x.
    boundary : str, optional
        Boundary condition type. One of:

        - ``"natural"``: Zero second derivative at endpoints.
        - ``"clamped"``: Specified first derivative at endpoints.
        - ``"not_a_knot"``: Third derivative continuity at second and
          second-to-last knots (default).
        - ``"periodic"``: Periodic boundary conditions.

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
    >>> x = torch.linspace(0, 1, 10)
    >>> y = torch.sin(x * 2 * torch.pi)
    >>> f = cubic_spline(x, y)
    >>> f(torch.tensor([0.5]))  # Evaluate at x=0.5
    """
    fitted = cubic_spline_fit(x, y, boundary=boundary, extrapolate=extrapolate)
    return lambda t: cubic_spline_evaluate(fitted, t)
