"""PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline."""

from typing import Callable

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from ._pchip_evaluate import pchip_evaluate
from ._pchip_fit import pchip_fit


@tensorclass
class PCHIPSpline:
    """Piecewise Cubic Hermite Interpolating Polynomial.

    PCHIP interpolation preserves monotonicity and avoids overshoot.
    Uses the Fritsch-Carlson algorithm for computing slopes.

    Attributes
    ----------
    knots : Tensor
        Breakpoints, shape (n_knots,). Strictly increasing.
    coefficients : Tensor
        Polynomial coefficients, shape (n_segments, 4, *value_shape).
        For segment i, the polynomial is:
        a[i] + b[i]*(t-knots[i]) + c[i]*(t-knots[i])^2 + d[i]*(t-knots[i])^3
        where coefficients[i] = [a, b, c, d].
    extrapolate : str
        Extrapolation mode: "error", "clamp", "extend".
    """

    knots: Tensor
    coefficients: Tensor
    extrapolate: str


def pchip(
    x: torch.Tensor,
    y: torch.Tensor,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a PCHIP interpolator from data.

    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves
    monotonicity of the data and avoids overshoot. It passes through
    all data points.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates. Must be strictly monotonically increasing.
    y : Tensor
        Data y-values. Shape must be compatible with x.
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
    >>> f = pchip(x, y)
    >>> f(torch.tensor([0.5]))  # Evaluate at x=0.5
    """
    fitted = pchip_fit(x, y, extrapolate=extrapolate)
    return lambda t: pchip_evaluate(fitted, t)
