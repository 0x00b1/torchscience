from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._derivative import derivative


def _curl_impl(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute curl of a 3D vector field.

    The curl is the vector field defined by:
    curl(V) = (dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)

    Parameters
    ----------
    vector_field : Tensor
        Input 3D vector field with shape (..., 3, nx, ny, nz).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute curl. Must have length 3.
        Default uses dimensions 1, 2, 3 (after the component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Curl field with shape (..., 3, nx, ny, nz).

    Examples
    --------
    >>> # Curl of gradient is zero
    >>> V = torch.randn(3, 10, 10, 10)
    >>> c = curl(V, dx=0.1)  # Shape: (3, 10, 10, 10)

    Raises
    ------
    ValueError
        If the vector field is not 3D (does not have exactly 3 components).
    """
    ndim = vector_field.ndim
    n_components = vector_field.shape[0]

    if n_components != 3:
        raise ValueError(
            f"Curl requires 3D vector field, got {n_components} components"
        )

    # Determine spatial dimensions
    if dim is None:
        spatial_dims = (1, 2, 3)
    else:
        spatial_dims = tuple(d if d >= 0 else ndim + d for d in dim)

    if len(spatial_dims) != 3:
        raise ValueError(
            f"Curl requires exactly 3 spatial dimensions, got {len(spatial_dims)}"
        )

    # Handle dx
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx), float(dx), float(dx))
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != 3:
            raise ValueError(
                f"dx must have 3 elements for 3D curl, got {len(dx_tuple)}"
            )

    # Extract components: Vx, Vy, Vz
    Vx = vector_field[0]  # Shape: (nx, ny, nz)
    Vy = vector_field[1]
    Vz = vector_field[2]

    # Spatial dims in components (shifted down by 1 since component dim is removed)
    dim_x = spatial_dims[0] - 1  # x dimension in component tensor
    dim_y = spatial_dims[1] - 1  # y dimension in component tensor
    dim_z = spatial_dims[2] - 1  # z dimension in component tensor

    dx_val, dy_val, dz_val = dx_tuple

    # Compute curl components:
    # curl_x = dVz/dy - dVy/dz
    dVz_dy = derivative(
        Vz, dim=dim_y, order=1, dx=dy_val, accuracy=accuracy, boundary=boundary
    )
    dVy_dz = derivative(
        Vy, dim=dim_z, order=1, dx=dz_val, accuracy=accuracy, boundary=boundary
    )
    curl_x = dVz_dy - dVy_dz

    # curl_y = dVx/dz - dVz/dx
    dVx_dz = derivative(
        Vx, dim=dim_z, order=1, dx=dz_val, accuracy=accuracy, boundary=boundary
    )
    dVz_dx = derivative(
        Vz, dim=dim_x, order=1, dx=dx_val, accuracy=accuracy, boundary=boundary
    )
    curl_y = dVx_dz - dVz_dx

    # curl_z = dVy/dx - dVx/dy
    dVy_dx = derivative(
        Vy, dim=dim_x, order=1, dx=dx_val, accuracy=accuracy, boundary=boundary
    )
    dVx_dy = derivative(
        Vx, dim=dim_y, order=1, dx=dy_val, accuracy=accuracy, boundary=boundary
    )
    curl_z = dVy_dx - dVx_dy

    return torch.stack([curl_x, curl_y, curl_z], dim=0)


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def curl(
    vector_field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
) -> Tensor:
    """Compute curl of a 3D vector field.

    The curl is the vector field defined by:
    curl(V) = (dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)

    Parameters
    ----------
    vector_field : Tensor
        Input 3D vector field with shape (..., 3, nx, ny, nz).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0.
    dim : tuple of int, optional
        Spatial dimensions over which to compute curl. Must have length 3.
        Default uses dimensions 1, 2, 3 (after the component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

    Returns
    -------
    Tensor
        Curl field with shape (..., 3, nx, ny, nz).

    Examples
    --------
    >>> # Curl of gradient is zero
    >>> V = torch.randn(3, 10, 10, 10)
    >>> c = curl(V, dx=0.1)  # Shape: (3, 10, 10, 10)

    Raises
    ------
    ValueError
        If the vector field is not 3D (does not have exactly 3 components).
    """
    return _curl_impl(vector_field, dx, dim, accuracy, boundary)
