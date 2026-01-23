"""Tensor product spline representation and convenience function."""

from typing import Callable

import torch
from tensordict.tensorclass import tensorclass
from torch import Tensor


@tensorclass
class TensorProductSpline:
    """Bicubic tensor product spline for 2D interpolation.

    A tensor product spline interpolates data on a rectangular grid
    using the product of 1D basis functions.

    Attributes
    ----------
    x_knots : Tensor
        Knot locations along x-axis, shape (nx,)
    y_knots : Tensor
        Knot locations along y-axis, shape (ny,)
    coefficients : Tensor
        Spline coefficients, shape (nx, ny, *value_shape)
        These are the function values at grid points.
    extrapolate : str
        How to handle out-of-domain queries: "error", "clamp", "extrapolate"
    """

    x_knots: Tensor
    y_knots: Tensor
    coefficients: Tensor
    extrapolate: str


def tensor_product_spline(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    extrapolate: str = "error",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a bicubic tensor product spline from gridded data.

    This is a convenience function that fits a tensor product spline and
    returns a callable that evaluates it.

    Parameters
    ----------
    x : Tensor
        Grid x-coordinates, shape (nx,). Must be strictly increasing.
    y : Tensor
        Grid y-coordinates, shape (ny,). Must be strictly increasing.
    z : Tensor
        Grid values, shape (nx, ny) or (nx, ny, *value_shape).
    extrapolate : str, optional
        How to handle out-of-domain queries. One of:

        - ``"error"``: Raise ExtrapolationError (default).
        - ``"clamp"``: Clamp to boundary values.
        - ``"extrapolate"``: Linear extrapolation from boundaries.

    Returns
    -------
    spline : Callable[[Tensor, Tensor], Tensor]
        Function that evaluates the spline at given (x, y) points.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 1, 10)
    >>> y = torch.linspace(0, 1, 10)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> Z = torch.sin(X * torch.pi) * torch.cos(Y * torch.pi)
    >>> f = tensor_product_spline(x, y, Z)
    >>> f(torch.tensor([0.5]), torch.tensor([0.5]))
    """
    from ._tensor_product_evaluate import tensor_product_evaluate
    from ._tensor_product_fit import tensor_product_fit

    spline = tensor_product_fit(x, y, z, extrapolate=extrapolate)
    return lambda qx, qy: tensor_product_evaluate(spline, qx, qy)
