"""Catmull-Rom spline representation and convenience function."""

from typing import Callable

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from ._catmull_rom_evaluate import catmull_rom_evaluate


@tensorclass
class CatmullRomSpline:
    """Catmull-Rom spline defined by control points.

    A Catmull-Rom spline passes through all interior control points
    (all points except the first and last for open curves).
    The tangent at each point is computed from neighboring points.

    Attributes
    ----------
    control_points : Tensor
        Control points, shape (n, *value_shape) where n >= 4.
        The spline passes through control_points[1:-1].
    alpha : float
        Parameterization type:
        - 0.0: Uniform (standard Catmull-Rom)
        - 0.5: Centripetal (recommended, avoids cusps and loops)
        - 1.0: Chordal (proportional to distance between points)
    extrapolate : str
        How to handle out-of-domain queries: "error", "clamp", "extrapolate"

    Notes
    -----
    The curve is defined on parameter range [0, n-3] for n control points.
    Each integer parameter value corresponds to a control point that the
    curve passes through.
    """

    control_points: Tensor
    alpha: float
    extrapolate: str


def catmull_rom(
    points: torch.Tensor,
    alpha: float = 0.5,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a Catmull-Rom spline from control points.

    This is a convenience function that creates a CatmullRomSpline and
    returns a callable that evaluates it.

    Parameters
    ----------
    points : Tensor
        Control points, shape (n, *value_shape) where n >= 4.
        The spline passes through points[1:-1].
    alpha : float, optional
        Parameterization type. Default is 0.5 (centripetal).

        - ``0.0``: Uniform (standard Catmull-Rom)
        - ``0.5``: Centripetal (recommended, avoids cusps and loops)
        - ``1.0``: Chordal (proportional to distance between points)

    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError (default).
        - ``"clamp"``: Clamp to boundary values.
        - ``"extrapolate"``: Allow extrapolation.

    Returns
    -------
    spline : Callable[[Tensor], Tensor]
        Function that evaluates the spline at given parameter values.

    Examples
    --------
    >>> import torch
    >>> # Create a smooth curve through 4 points
    >>> points = torch.tensor([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., 1.]
    ... ])
    >>> curve = catmull_rom(points)
    >>> # Evaluate at parameter 0.5 (between first two interior points)
    >>> curve(torch.tensor([0.5]))
    """
    if points.shape[0] < 4:
        raise ValueError(
            f"Catmull-Rom spline requires at least 4 control points, got {points.shape[0]}"
        )

    spline = CatmullRomSpline(
        control_points=points,
        alpha=alpha,
        extrapolate=extrapolate,
        batch_size=[],
    )
    return lambda t: catmull_rom_evaluate(spline, t)
