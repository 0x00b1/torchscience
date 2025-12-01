from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation._derivative import derivative


def laplacian(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Laplacian of a scalar field.

    The Laplacian is the sum of second partial derivatives: nabla^2 f = sum_i d^2f/dx_i^2.

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute Laplacian. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Laplacian field with the same shape as the input.

    Examples
    --------
    >>> # Laplacian of x^2 + y^2 is 4
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> f = X**2 + Y**2
    >>> lap = laplacian(f, dx=0.05)  # Should be ~4
    """
    ndim = field.ndim

    # Determine which dimensions to differentiate
    if dim is None:
        spatial_dims = tuple(range(ndim))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Sum second derivatives
    result = torch.zeros_like(field)
    for i, d in enumerate(spatial_dims):
        second_deriv = derivative(
            field,
            dim=d,
            order=2,
            dx=dx_tuple[i],
            accuracy=accuracy,
            kind="central",
            boundary=boundary,
        )
        result = result + second_deriv

    return result
