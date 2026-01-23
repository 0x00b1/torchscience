"""Wave operator for wave equation simulations."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._grid import IrregularMesh, RegularGrid
from torchscience.differentiation._laplacian import laplacian


def _wave_operator_impl(
    field: Tensor,
    wave_speed: Union[float, Tensor],
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute wave operator c^2 * laplacian(f).

    This is the spatial part of the wave equation d^2f/dt^2 = c^2 * laplacian(f).

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    wave_speed : float or Tensor
        Wave propagation speed. Scalar for constant, or tensor for
        spatially varying speed (heterogeneous media).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions over which to compute the wave operator. Default uses
        all dimensions.
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
        Wave operator c^2 * laplacian(f) with shape (*spatial).

    Notes
    -----
    The wave equation is:
        d^2f/dt^2 = c^2 * laplacian(f)

    This function computes the right-hand side, which can be used in
    time-stepping schemes to evolve wave fields.

    For variable wave speed (heterogeneous media), the wave equation
    becomes more complex, but c^2 * laplacian(f) remains a useful
    approximation for slowly varying c.

    Examples
    --------
    >>> # Acoustic wave in 2D
    >>> pressure = torch.randn(64, 64)
    >>> c = 343.0  # speed of sound in m/s
    >>> dx = 0.01  # 1 cm grid spacing
    >>> wave_term = wave_operator(pressure, wave_speed=c, dx=dx)
    """
    # Handle sparse tensors by densifying
    if field.is_sparse:
        field = field.to_dense()

    # If grid is provided, extract spacing and boundary from it
    if grid is not None:
        if isinstance(grid, RegularGrid):
            dx = tuple(grid.spacing.tolist())
            boundary = grid.boundary
        else:
            raise NotImplementedError(
                "IrregularMesh support for wave_operator not yet implemented"
            )

    # Compute Laplacian
    lap = laplacian(
        field, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
    )

    # Multiply by c^2
    if isinstance(wave_speed, (int, float)):
        return wave_speed**2 * lap
    else:
        # Handle sparse wave_speed tensor
        if wave_speed.is_sparse:
            wave_speed = wave_speed.to_dense()
        return wave_speed**2 * lap


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def wave_operator(
    field: Tensor,
    wave_speed: Union[float, Tensor],
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute wave operator c^2 * laplacian(f).

    This is the spatial part of the wave equation d^2f/dt^2 = c^2 * laplacian(f).

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    wave_speed : float or Tensor
        Wave propagation speed. Scalar for constant, or tensor for
        spatially varying speed (heterogeneous media).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions over which to compute the wave operator. Default uses
        all dimensions.
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
        Wave operator c^2 * laplacian(f) with shape (*spatial).

    Notes
    -----
    The wave equation is:
        d^2f/dt^2 = c^2 * laplacian(f)

    This function computes the right-hand side, which can be used in
    time-stepping schemes to evolve wave fields.

    For variable wave speed (heterogeneous media), the wave equation
    becomes more complex, but c^2 * laplacian(f) remains a useful
    approximation for slowly varying c.

    Examples
    --------
    >>> # Acoustic wave in 2D
    >>> pressure = torch.randn(64, 64)
    >>> c = 343.0  # speed of sound in m/s
    >>> dx = 0.01  # 1 cm grid spacing
    >>> wave_term = wave_operator(pressure, wave_speed=c, dx=dx)
    """
    return _wave_operator_impl(
        field, wave_speed, dx, dim, accuracy, boundary, grid=grid
    )
