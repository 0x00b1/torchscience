"""Vorticity computation for fluid dynamics."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._curl import curl
from torchscience.differentiation._derivative import derivative
from torchscience.differentiation._grid import IrregularMesh, RegularGrid


def _vorticity_impl(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute vorticity of a velocity field.

    Vorticity is the curl of velocity: omega = nabla x v

    For 2D: omega = dv_y/dx - dv_x/dy (scalar)
    For 3D: omega = (dv_z/dy - dv_y/dz, dv_x/dz - dv_z/dx, dv_y/dx - dv_x/dy)

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (ndim, *spatial) where ndim is 2 or 3.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions (default: last ndim dims after component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        dx and boundary parameters.

    Returns
    -------
    Tensor
        Vorticity field.
        - 2D: scalar field with shape (*spatial)
        - 3D: vector field with shape (3, *spatial)

    Examples
    --------
    >>> # Rigid body rotation in 2D
    >>> x = torch.linspace(-1, 1, 32)
    >>> X, Y = torch.meshgrid(x, x, indexing='ij')
    >>> velocity = torch.stack([-Y, X], dim=0)
    >>> omega = vorticity(velocity, dx=2/31)
    >>> # omega approximately equals 2 everywhere
    """
    # Handle sparse tensors by densifying
    if velocity.is_sparse:
        velocity = velocity.to_dense()

    # If grid is provided, extract spacing and boundary from it
    if grid is not None:
        if isinstance(grid, RegularGrid):
            dx = tuple(grid.spacing.tolist())
            boundary = grid.boundary
        else:
            raise NotImplementedError(
                "IrregularMesh support for vorticity not yet implemented"
            )

    ndim = velocity.shape[0]

    if ndim == 2:
        # 2D: omega = dv_y/dx - dv_x/dy
        spatial_ndim = velocity.ndim - 1
        if dim is None:
            # Default: component dim is 0, spatial dims are 1, 2
            # After extracting component, spatial dims are 0, 1
            dim_x = 0  # x dimension in component tensor (corresponds to dim 1)
            dim_y = 1  # y dimension in component tensor (corresponds to dim 2)
        else:
            # User-specified dims (relative to full velocity tensor)
            norm_dim = tuple(d if d >= 0 else velocity.ndim + d for d in dim)
            # Adjust for component dimension removal
            dim_x = norm_dim[0] - 1 if norm_dim[0] > 0 else norm_dim[0]
            dim_y = norm_dim[1] - 1 if norm_dim[1] > 0 else norm_dim[1]

        # Handle dx
        if isinstance(dx, (int, float)):
            dx_val = float(dx)
            dy_val = float(dx)
        elif isinstance(dx, Tensor):
            dx_list = dx.tolist()
            dx_val = dx_list[0] if len(dx_list) > 0 else 1.0
            dy_val = dx_list[1] if len(dx_list) > 1 else dx_val
        else:
            dx_tuple = tuple(float(d) for d in dx)
            dx_val = dx_tuple[0] if len(dx_tuple) > 0 else 1.0
            dy_val = dx_tuple[1] if len(dx_tuple) > 1 else dx_val

        # Extract velocity components
        vx = velocity[0]  # v_x component
        vy = velocity[1]  # v_y component

        # Compute partial derivatives
        dvx_dy = derivative(
            vx,
            dim=dim_y,
            order=1,
            dx=dy_val,
            accuracy=accuracy,
            boundary=boundary,
        )
        dvy_dx = derivative(
            vy,
            dim=dim_x,
            order=1,
            dx=dx_val,
            accuracy=accuracy,
            boundary=boundary,
        )

        return dvy_dx - dvx_dy

    elif ndim == 3:
        # 3D: omega = curl(v)
        return curl(
            velocity,
            dx=dx,
            dim=dim,
            accuracy=accuracy,
            boundary=boundary,
            grid=grid,
        )

    else:
        raise ValueError(
            f"Vorticity requires 2D or 3D velocity field, got {ndim} components"
        )


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def vorticity(
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute vorticity of a velocity field.

    Vorticity is the curl of velocity: omega = nabla x v

    For 2D: omega = dv_y/dx - dv_x/dy (scalar)
    For 3D: omega = (dv_z/dy - dv_y/dz, dv_x/dz - dv_z/dx, dv_y/dx - dv_x/dy)

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (ndim, *spatial) where ndim is 2 or 3.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions (default: last ndim dims after component dimension).
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        dx and boundary parameters.

    Returns
    -------
    Tensor
        Vorticity field.
        - 2D: scalar field with shape (*spatial)
        - 3D: vector field with shape (3, *spatial)

    Examples
    --------
    >>> # Rigid body rotation in 2D
    >>> x = torch.linspace(-1, 1, 32)
    >>> X, Y = torch.meshgrid(x, x, indexing='ij')
    >>> velocity = torch.stack([-Y, X], dim=0)
    >>> omega = vorticity(velocity, dx=2/31)
    >>> # omega approximately equals 2 everywhere
    """
    return _vorticity_impl(velocity, dx, dim, accuracy, boundary, grid=grid)
