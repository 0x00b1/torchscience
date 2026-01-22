from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._derivative import derivative


def _gradient_impl(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute gradient of a scalar field.

    The gradient is the vector of partial derivatives along each spatial dimension.

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute gradient. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Gradient field. If dim is None or specifies all dimensions, shape is
        (ndim, *field.shape). If dim specifies a subset, shape is
        (...batch_dims, len(dim), *spatial_dims) where the gradient components
        are inserted before the spatial dimensions.

    Examples
    --------
    >>> # 2D field
    >>> f = torch.randn(20, 30)
    >>> grad = gradient(f, dx=0.1)  # Shape: (2, 20, 30)

    >>> # Batched field with gradient over last 2 dimensions
    >>> f = torch.randn(10, 20, 30)
    >>> grad = gradient(f, dim=(-2, -1), dx=0.1)  # Shape: (10, 2, 20, 30)
    """
    # Internal implementation - see gradient() for docstring
    ndim = field.ndim

    # Determine which dimensions to differentiate
    if dim is None:
        # All dimensions
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

    # Compute partial derivatives
    partials = []
    for i, d in enumerate(spatial_dims):
        partial = derivative(
            field,
            dim=d,
            order=1,
            dx=dx_tuple[i],
            accuracy=accuracy,
            kind="central",
            boundary=boundary,
        )
        partials.append(partial)

    # Stack results
    # If dim is None (all dims), result is (ndim, *field.shape)
    # If dim specifies subset, insert gradient components before spatial dims
    if dim is None:
        result = torch.stack(partials, dim=0)
    else:
        # Find where to insert the gradient dimension
        # Insert before the first spatial dimension
        min_spatial_dim = min(spatial_dims)
        result = torch.stack(partials, dim=min_spatial_dim)

    return result


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def gradient(
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute gradient of a scalar field.

    The gradient is the vector of partial derivatives along each spatial dimension.

    Parameters
    ----------
    field : Tensor
        Input scalar field with shape (..., *spatial_dims).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute gradient. Default uses all dimensions.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Gradient field. If dim is None or specifies all dimensions, shape is
        (ndim, *field.shape). If dim specifies a subset, shape is
        (...batch_dims, len(dim), *spatial_dims) where the gradient components
        are inserted before the spatial dimensions.

    Examples
    --------
    >>> # 2D field
    >>> f = torch.randn(20, 30)
    >>> grad = gradient(f, dx=0.1)  # Shape: (2, 20, 30)

    >>> # Batched field with gradient over last 2 dimensions
    >>> f = torch.randn(10, 20, 30)
    >>> grad = gradient(f, dim=(-2, -1), dx=0.1)  # Shape: (10, 2, 20, 30)
    """
    return _gradient_impl(field, dx, dim, accuracy, boundary)
