from __future__ import annotations

from typing import Tuple, Union

from torch import Tensor

from torchscience.differentiation._derivative import derivative


def divergence(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute divergence of a vector field.

    The divergence is the sum of partial derivatives of each component with respect
    to its corresponding coordinate: div(V) = sum_i dV_i/dx_i.

    Parameters
    ----------
    vector_field : Tensor
        Input vector field with shape (..., ndim, *spatial_dims) where the
        component dimension contains the vector components and spatial_dims
        are the spatial dimensions.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute divergence. Default uses all
        dimensions after the component dimension.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Divergence field with shape (..., *spatial_dims).

    Examples
    --------
    >>> # Divergence of (x, y) is 2
    >>> x = torch.linspace(0, 1, 21)
    >>> y = torch.linspace(0, 1, 21)
    >>> X, Y = torch.meshgrid(x, y, indexing="ij")
    >>> V = torch.stack([X, Y], dim=0)  # Shape: (2, 21, 21)
    >>> div = divergence(V, dx=0.05)  # Shape: (21, 21)
    """
    ndim = vector_field.ndim

    # Assume first dimension is the component dimension (by default)
    # The number of components determines the number of spatial dimensions
    n_components = vector_field.shape[0]

    # Determine spatial dimensions
    if dim is None:
        # Use dimensions 1, 2, ..., n_components as spatial dims
        spatial_dims = tuple(range(1, n_components + 1))
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    n_spatial = len(spatial_dims)

    if n_components != n_spatial:
        raise ValueError(
            f"Number of vector components ({n_components}) must match "
            f"number of spatial dimensions ({n_spatial})"
        )

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * n_spatial
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != n_spatial:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but {n_spatial} spatial dimensions"
            )

    # Compute divergence: sum of dV_i/dx_i
    result = None
    for i in range(n_components):
        # Get component i: shape (..., *spatial_dims)
        component = vector_field[i]

        # When we index vector_field[i], the component dimension is removed,
        # so spatial dims are shifted down by 1
        spatial_dim_in_component = spatial_dims[i] - 1

        # Compute derivative of component i w.r.t. coordinate i
        partial = derivative(
            component,
            dim=spatial_dim_in_component,
            order=1,
            dx=dx_tuple[i],
            accuracy=accuracy,
            kind="central",
            boundary=boundary,
        )

        if result is None:
            result = partial
        else:
            result = result + partial

    return result
