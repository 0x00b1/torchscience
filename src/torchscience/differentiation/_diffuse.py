"""Diffusion operator for heat/mass transfer."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._divergence import divergence
from torchscience.differentiation._gradient import gradient
from torchscience.differentiation._grid import IrregularMesh, RegularGrid
from torchscience.differentiation._laplacian import laplacian


def _diffuse_impl(
    field: Tensor,
    diffusivity: Union[float, Tensor],
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute diffusion term div(D grad f).

    For constant D, this simplifies to D * laplacian(f).
    For spatially varying D, computes the full divergence form.

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    diffusivity : float or Tensor
        Diffusion coefficient. Scalar for constant, or tensor for
        spatially varying diffusivity.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions over which to compute diffusion. Default uses all dimensions.
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
        Diffusion term with shape (*spatial).

    Notes
    -----
    For constant diffusivity:
        div(D grad f) = D * laplacian(f)

    For variable diffusivity:
        div(D grad f) = D * laplacian(f) + grad(D) . grad(f)

    The diffusion equation df/dt = div(D grad f) describes heat conduction,
    mass diffusion, and other transport phenomena.

    Examples
    --------
    >>> # Heat diffusion with constant thermal diffusivity
    >>> temperature = torch.randn(32, 32)
    >>> alpha = 0.01  # thermal diffusivity
    >>> heat_flux = diffuse(temperature, diffusivity=alpha, dx=0.1)
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
                "IrregularMesh support for diffuse not yet implemented"
            )

    if isinstance(diffusivity, (int, float)):
        # Constant diffusivity: D * laplacian(f)
        lap = laplacian(
            field, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
        )
        return diffusivity * lap
    else:
        # Variable diffusivity: div(D * grad(f))
        # Handle sparse diffusivity tensor
        if diffusivity.is_sparse:
            diffusivity = diffusivity.to_dense()

        # Compute gradient of field: shape (ndim, *spatial)
        grad_f = gradient(
            field, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
        )

        # Compute D * grad(f): shape (ndim, *spatial)
        # Expand diffusivity to broadcast with gradient components
        D_grad_f = diffusivity * grad_f

        # Compute divergence: div(D * grad(f))
        return divergence(
            D_grad_f, dx=dx, dim=dim, accuracy=accuracy, boundary=boundary
        )


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def diffuse(
    field: Tensor,
    diffusivity: Union[float, Tensor],
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute diffusion term div(D grad f).

    For constant D, this simplifies to D * laplacian(f).
    For spatially varying D, computes the full divergence form.

    Parameters
    ----------
    field : Tensor
        Scalar field with shape (*spatial).
    diffusivity : float or Tensor
        Diffusion coefficient. Scalar for constant, or tensor for
        spatially varying diffusivity.
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions over which to compute diffusion. Default uses all dimensions.
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
        Diffusion term with shape (*spatial).

    Notes
    -----
    For constant diffusivity:
        div(D grad f) = D * laplacian(f)

    For variable diffusivity:
        div(D grad f) = D * laplacian(f) + grad(D) . grad(f)

    The diffusion equation df/dt = div(D grad f) describes heat conduction,
    mass diffusion, and other transport phenomena.

    Examples
    --------
    >>> # Heat diffusion with constant thermal diffusivity
    >>> temperature = torch.randn(32, 32)
    >>> alpha = 0.01  # thermal diffusivity
    >>> heat_flux = diffuse(temperature, diffusivity=alpha, dx=0.1)
    """
    return _diffuse_impl(
        field, diffusivity, dx, dim, accuracy, boundary, grid=grid
    )
