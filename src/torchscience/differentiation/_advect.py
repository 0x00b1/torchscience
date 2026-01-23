"""Advection operator for CFD."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._gradient import gradient
from torchscience.differentiation._grid import IrregularMesh, RegularGrid


def _advect_impl(
    field: Tensor,
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute advection term (v . grad)f.

    The advection term represents the rate of change of a quantity
    due to transport by the flow.

    Parameters
    ----------
    field : Tensor
        Scalar field to advect with shape (*spatial).
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions (default: all dims of field).
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
        Advection term with shape (*spatial).

    Notes
    -----
    The advection term is computed as:

        (v . grad)f = v_x * df/dx + v_y * df/dy + v_z * df/dz

    This uses central differences. For stability in time integration,
    consider upwind schemes for hyperbolic problems.

    Examples
    --------
    >>> # Advection in 2D
    >>> field = torch.randn(32, 32)
    >>> velocity = torch.randn(2, 32, 32)
    >>> advection = advect(field, velocity, dx=0.1)
    """
    # Handle sparse tensors by densifying
    if field.is_sparse:
        field = field.to_dense()
    if velocity.is_sparse:
        velocity = velocity.to_dense()

    # If grid is provided, extract spacing and boundary from it
    if grid is not None:
        if isinstance(grid, RegularGrid):
            dx = tuple(grid.spacing.tolist())
            boundary = grid.boundary
        else:
            raise NotImplementedError(
                "IrregularMesh support for advect not yet implemented"
            )

    ndim = velocity.shape[0]

    if dim is None:
        dim = tuple(range(-ndim, 0))

    # Compute gradient of field
    grad_f = gradient(
        field, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )

    # Dot product: v . grad f = sum_i v_i * df/dx_i
    advection = (velocity * grad_f).sum(dim=0)

    return advection


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def advect(
    field: Tensor,
    velocity: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute advection term (v . grad)f.

    The advection term represents the rate of change of a quantity
    due to transport by the flow.

    Parameters
    ----------
    field : Tensor
        Scalar field to advect with shape (*spatial).
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions (default: all dims of field).
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
        Advection term with shape (*spatial).

    Notes
    -----
    The advection term is computed as:

        (v . grad)f = v_x * df/dx + v_y * df/dy + v_z * df/dz

    This uses central differences. For stability in time integration,
    consider upwind schemes for hyperbolic problems.

    Examples
    --------
    >>> # Advection in 2D
    >>> field = torch.randn(32, 32)
    >>> velocity = torch.randn(2, 32, 32)
    >>> advection = advect(field, velocity, dx=0.1)
    """
    return _advect_impl(
        field, velocity, dx, dim, accuracy, boundary, grid=grid
    )
