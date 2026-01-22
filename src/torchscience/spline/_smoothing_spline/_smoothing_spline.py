"""Smoothing spline representation and convenience function."""

from typing import Callable, Optional

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class SmoothingSpline:
    """Cubic smoothing spline representation.

    A smoothing spline minimizes a weighted combination of fitting error
    and curvature penalty:

    Σ w_i (y_i - f(x_i))² + λ ∫ f''(x)² dx

    where λ is the smoothing parameter.

    Attributes
    ----------
    knots : Tensor
        Knot locations, shape (n,)
    coefficients : Tensor
        Spline coefficients, shape (n, *value_shape)
    smoothing : float
        Smoothing parameter (λ). Higher values produce smoother curves.
    extrapolate : str
        How to handle out-of-domain queries: "error", "clamp", "extrapolate"
    """

    knots: Tensor
    coefficients: Tensor
    smoothing: float
    extrapolate: str


def smoothing_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    smoothing: Optional[float] = None,
    weights: Optional[torch.Tensor] = None,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a smoothing spline from data.

    This is a convenience function that fits a smoothing spline and returns
    a callable that evaluates it.

    Parameters
    ----------
    x : Tensor
        Data x-coordinates, shape (n,). Must be strictly increasing.
    y : Tensor
        Data y-values, shape (n, *value_shape).
    smoothing : float, optional
        Smoothing parameter (λ). If None, automatic selection via GCV is used.
        Higher values produce smoother curves. Default is None.
    weights : Tensor, optional
        Weights for data points, shape (n,). Higher weight means the spline
        will fit that point more closely. Default is uniform weights.
    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError (default).
        - ``"clamp"``: Clamp to boundary values.
        - ``"extrapolate"``: Linear extrapolation from boundaries.

    Returns
    -------
    spline : Callable[[Tensor], Tensor]
        Function that evaluates the spline at given points.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 2*torch.pi, 50)
    >>> y = torch.sin(x) + 0.1 * torch.randn_like(x)  # Noisy sine
    >>> f = smoothing_spline(x, y)
    >>> f(torch.tensor([0.5, 1.0, 1.5]))

    Notes
    -----
    The smoothing spline minimizes:

    Σ w_i (y_i - f(x_i))² + λ ∫ f''(x)² dx

    - λ = 0: Interpolating spline (passes through all points)
    - λ → ∞: Linear regression (straight line)
    """
    from ._smoothing_spline_evaluate import smoothing_spline_evaluate
    from ._smoothing_spline_fit import smoothing_spline_fit

    spline = smoothing_spline_fit(
        x, y, smoothing=smoothing, weights=weights, extrapolate=extrapolate
    )
    return lambda t: smoothing_spline_evaluate(spline, t)
