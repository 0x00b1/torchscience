from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from torchscience.differentiation._derivative import derivative


def jacobian(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute Jacobian matrix of a vector field.

    The Jacobian is the matrix of all partial derivatives:
    J[i, j] = dV_i / dx_j.

    Parameters
    ----------
    vector_field : Tensor
        Input vector field with shape (..., m, *spatial_dims) where m is the
        number of vector components.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute Jacobian. Default uses dimensions
        1, 2, ..., n after the component dimension.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Jacobian field with shape (..., m, ndim, *spatial_dims) where m is the
        number of components and ndim is the number of spatial dimensions.

    Examples
    --------
    >>> # Jacobian of (2x + 3y, 4x + 5y) is [[2, 3], [4, 5]]
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> V = torch.stack([2*X + 3*Y, 4*X + 5*Y], dim=0)  # Shape: (2, 21, 21)
    >>> J = jacobian(V, dx=0.05)  # Shape: (2, 2, 21, 21)
    """
    ndim = vector_field.ndim
    n_components = vector_field.shape[0]

    # Determine spatial dimensions
    if dim is None:
        # Infer spatial dimensions based on tensor shape
        # Assume first dimension is components, rest are spatial
        n_spatial = ndim - 1
        spatial_dims = tuple(range(1, ndim))
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

    # Compute Jacobian: J[i, j] = dV_i / dx_j
    jacobian_rows = []

    for i in range(n_components):
        # Get component i: shape (*spatial_dims)
        component = vector_field[i]

        row = []
        for j, spatial_dim in enumerate(spatial_dims):
            # When we index vector_field[i], the component dimension is removed,
            # so spatial dims are shifted down by 1
            dim_in_component = spatial_dim - 1

            # Compute dV_i / dx_j
            partial = derivative(
                component,
                dim=dim_in_component,
                order=1,
                dx=dx_tuple[j],
                accuracy=accuracy,
                kind="central",
                boundary=boundary,
            )
            row.append(partial)

        # Stack row: shape (n_spatial, *spatial_dims)
        jacobian_rows.append(torch.stack(row, dim=0))

    # Stack all rows: shape (n_components, n_spatial, *spatial_dims)
    result = torch.stack(jacobian_rows, dim=0)

    return result
