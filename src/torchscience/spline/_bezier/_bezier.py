"""Bezier curve representation and convenience function."""

from typing import Callable

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor

from ._bezier_evaluate import bezier_evaluate


@tensorclass
class BezierCurve:
    """Bezier curve defined by control points.

    A Bezier curve of degree n is defined by n+1 control points.
    The curve parameter t ranges from 0 to 1, with:
    - t=0 corresponding to the first control point
    - t=1 corresponding to the last control point

    Attributes
    ----------
    control_points : Tensor
        Control points, shape (n+1, *value_shape) for degree n
    extrapolate : str
        How to handle out-of-domain queries: "error", "clamp", "extrapolate"
    """

    control_points: Tensor
    extrapolate: str

    @property
    def degree(self) -> int:
        """Return the degree of the Bezier curve."""
        return self.control_points.shape[0] - 1


def bezier(
    control_points: torch.Tensor,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a Bezier curve from control points.

    This is a convenience function that creates a BezierCurve and returns
    a callable that evaluates it.

    Parameters
    ----------
    control_points : Tensor
        Control points, shape (n+1, *value_shape) for degree n.
        For a 2D curve, shape would be (n+1, 2).
    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError for t outside [0, 1] (default).
        - ``"clamp"``: Clamp t to [0, 1].
        - ``"extrapolate"``: Allow extrapolation outside [0, 1].

    Returns
    -------
    curve : Callable[[Tensor], Tensor]
        Function that evaluates the Bezier curve at given parameter values.

    Examples
    --------
    >>> import torch
    >>> # Quadratic Bezier curve (degree 2) in 2D
    >>> control_points = torch.tensor([[0., 0.], [0.5, 1.], [1., 0.]])
    >>> curve = bezier(control_points)
    >>> curve(torch.tensor([0.0, 0.5, 1.0]))
    tensor([[0.0000, 0.0000],
            [0.5000, 0.5000],
            [1.0000, 0.0000]])
    """
    curve = BezierCurve(
        control_points=control_points,
        extrapolate=extrapolate,
        batch_size=[],
    )
    return lambda t: bezier_evaluate(curve, t)
