from typing import Callable, Optional

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from ._b_spline_evaluate import b_spline_evaluate
from ._b_spline_fit import b_spline_fit


@tensorclass
class BSpline:
    """B-spline representation with knots and control points.

    Attributes
    ----------
    knots : Tensor
        Knot vector, shape (n_knots,). Non-decreasing.
    control_points : Tensor
        Control points, shape (n_control, *y_dim) where n_control = n_knots - degree - 1
    degree : int
        Polynomial degree (stored as metadata, not tensor)
    extrapolate : str
        How to handle out-of-domain queries: "error", "clamp", "extrapolate"
    """

    knots: Tensor
    control_points: Tensor
    degree: int
    extrapolate: str


def b_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    n_knots: Optional[int] = None,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a B-spline approximation from data.

    This is a convenience function that fits a B-spline and returns
    a callable that evaluates it.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates. Must be strictly monotonically increasing.
    y : Tensor
        Data y-values. Shape must be compatible with x.
    degree : int, optional
        Spline degree. Default is 3 (cubic).
    n_knots : int, optional
        Number of interior knots. If not specified, a reasonable default
        is computed based on the data size.
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
    >>> x = torch.linspace(0, 1, 20)
    >>> y = torch.sin(x * 2 * torch.pi)
    >>> f = b_spline(x, y, degree=3, n_knots=5)
    >>> f(torch.tensor([0.5]))  # Evaluate at x=0.5
    """
    fitted = b_spline_fit(
        x, y, degree=degree, n_knots=n_knots, extrapolate=extrapolate
    )
    return lambda t: b_spline_evaluate(fitted, t)
