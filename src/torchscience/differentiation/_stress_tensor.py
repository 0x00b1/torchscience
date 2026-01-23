"""Stress tensor computation for fluid mechanics."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._grid import IrregularMesh, RegularGrid
from torchscience.differentiation._jacobian import jacobian


def _stress_tensor_impl(
    velocity: Tensor,
    pressure: Tensor,
    viscosity: float | Tensor,
    spacing: Union[float, Tuple[float, ...]] = 1.0,
    *,
    dims: Sequence[int] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute Cauchy stress tensor for Newtonian fluid.

    sigma_ij = -p*delta_ij + mu*(dv_i/dx_j + dv_j/dx_i)

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    pressure : Tensor
        Pressure field with shape (*spatial).
    viscosity : float or Tensor
        Dynamic viscosity mu. Scalar for constant, or tensor for
        spatially varying viscosity.
    spacing : float or sequence of floats
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dims : sequence of int, optional
        Spatial dimensions over which to compute derivatives. Default uses
        dimensions 1, 2, ..., n after the component dimension.
    accuracy : int
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        spacing and boundary parameters.

    Returns
    -------
    Tensor
        Stress tensor with shape (ndim, ndim, *spatial).

    Notes
    -----
    The Cauchy stress tensor for an incompressible Newtonian fluid is:

        sigma_ij = -p*delta_ij + tau_ij

    where tau_ij = mu*(dv_i/dx_j + dv_j/dx_i) is the viscous stress tensor.

    The diagonal terms (-p + 2*mu*dv_i/dx_i) represent normal stresses.
    The off-diagonal terms (mu*(dv_i/dx_j + dv_j/dx_i)) represent shear stresses.

    Examples
    --------
    >>> # Stress in viscous flow
    >>> velocity = torch.randn(3, 32, 32, 32)
    >>> pressure = torch.randn(32, 32, 32)
    >>> mu = 1e-3  # dynamic viscosity of water
    >>> sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=0.01)
    """
    # Handle sparse tensors by densifying
    if velocity.is_sparse:
        velocity = velocity.to_dense()
    if pressure.is_sparse:
        pressure = pressure.to_dense()

    ndim = velocity.shape[0]
    spatial_shape = velocity.shape[1:]

    # Compute velocity gradient tensor: J_ij = dv_i/dx_j
    J = jacobian(
        velocity,
        dx=spacing,
        dim=dims,
        accuracy=accuracy,
        boundary=boundary,
        grid=grid,
    )

    # Viscous stress: tau_ij = mu*(J_ij + J_ji) = mu*(dv_i/dx_j + dv_j/dx_i)
    tau = viscosity * (J + J.transpose(0, 1))

    # Pressure contribution: -p*delta_ij
    # Create identity tensor for each spatial point
    identity = torch.zeros(
        ndim,
        ndim,
        *spatial_shape,
        device=velocity.device,
        dtype=velocity.dtype,
    )
    for i in range(ndim):
        identity[i, i] = 1.0

    # Stress tensor: sigma_ij = -p*delta_ij + tau_ij
    sigma = -pressure * identity + tau

    return sigma


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def stress_tensor(
    velocity: Tensor,
    pressure: Tensor,
    viscosity: float | Tensor,
    spacing: Union[float, Tuple[float, ...]] = 1.0,
    *,
    dims: Sequence[int] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute Cauchy stress tensor for Newtonian fluid.

    sigma_ij = -p*delta_ij + mu*(dv_i/dx_j + dv_j/dx_i)

    Parameters
    ----------
    velocity : Tensor
        Velocity field with shape (ndim, *spatial).
    pressure : Tensor
        Pressure field with shape (*spatial).
    viscosity : float or Tensor
        Dynamic viscosity mu. Scalar for constant, or tensor for
        spatially varying viscosity.
    spacing : float or sequence of floats
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dims : sequence of int, optional
        Spatial dimensions over which to compute derivatives. Default uses
        dimensions 1, 2, ..., n after the component dimension.
    accuracy : int
        Accuracy order of the finite difference approximation. Default is 2.
    boundary : str
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        spacing and boundary parameters.

    Returns
    -------
    Tensor
        Stress tensor with shape (ndim, ndim, *spatial).

    Notes
    -----
    The Cauchy stress tensor for an incompressible Newtonian fluid is:

        sigma_ij = -p*delta_ij + tau_ij

    where tau_ij = mu*(dv_i/dx_j + dv_j/dx_i) is the viscous stress tensor.

    The diagonal terms (-p + 2*mu*dv_i/dx_i) represent normal stresses.
    The off-diagonal terms (mu*(dv_i/dx_j + dv_j/dx_i)) represent shear stresses.

    Examples
    --------
    >>> # Stress in viscous flow
    >>> velocity = torch.randn(3, 32, 32, 32)
    >>> pressure = torch.randn(32, 32, 32)
    >>> mu = 1e-3  # dynamic viscosity of water
    >>> sigma = stress_tensor(velocity, pressure, viscosity=mu, spacing=0.01)
    """
    return _stress_tensor_impl(
        velocity,
        pressure,
        viscosity,
        spacing,
        dims=dims,
        accuracy=accuracy,
        boundary=boundary,
        grid=grid,
    )
