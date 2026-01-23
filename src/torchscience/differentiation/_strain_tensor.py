"""Strain tensor computation for solid mechanics."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._grid import IrregularMesh, RegularGrid
from torchscience.differentiation._jacobian import jacobian


def _strain_tensor_impl(
    displacement: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute symmetric strain tensor from displacement field.

    The infinitesimal strain tensor is defined as:
    epsilon_ij = 1/2 * (du_i/dx_j + du_j/dx_i)

    Parameters
    ----------
    displacement : Tensor
        Displacement field with shape (ndim, *spatial) where ndim is the
        number of spatial dimensions (2 or 3).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions over which to compute derivatives. Default uses
        dimensions 1, 2, ..., n after the component dimension.
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
        Strain tensor with shape (ndim, ndim, *spatial).

    Notes
    -----
    This computes the infinitesimal (linear) strain tensor, which is
    valid for small deformations. For large deformations, consider
    the Green-Lagrange or Almansi strain tensors.

    The strain tensor is symmetric: epsilon_ij = epsilon_ji.

    Diagonal components (epsilon_ii) represent normal strains (extension/compression).
    Off-diagonal components (epsilon_ij, i != j) represent shear strains.

    The volumetric strain (dilatation) is the trace: theta = epsilon_11 + epsilon_22 + epsilon_33
    The deviatoric strain tensor is: e_ij = epsilon_ij - (1/3) * theta * delta_ij

    Examples
    --------
    >>> # Uniform extension in x-direction
    >>> n = 32
    >>> x = torch.linspace(0, 1, n)
    >>> X, Y = torch.meshgrid(x, x, indexing='ij')
    >>> displacement = torch.stack([0.1 * X, 0.0 * Y], dim=0)
    >>> eps = strain_tensor(displacement, dx=1/(n-1))
    >>> eps[0, 0].mean()  # approximately 0.1
    """
    # Handle sparse tensors by densifying
    if displacement.is_sparse:
        displacement = displacement.to_dense()

    # Compute displacement gradient: J_ij = du_i/dx_j
    J = jacobian(
        displacement,
        dx=dx,
        dim=dim,
        accuracy=accuracy,
        boundary=boundary,
        grid=grid,
    )

    # Symmetric strain tensor: epsilon = 1/2 * (J + J^T)
    eps = 0.5 * (J + J.transpose(0, 1))

    return eps


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def strain_tensor(
    displacement: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    accuracy: int = 2,
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute symmetric strain tensor from displacement field.

    The infinitesimal strain tensor is defined as:
    epsilon_ij = 1/2 * (du_i/dx_j + du_j/dx_i)

    Parameters
    ----------
    displacement : Tensor
        Displacement field with shape (ndim, *spatial) where ndim is the
        number of spatial dimensions (2 or 3).
    dx : float or tuple of float, optional
        Grid spacing. Scalar applies to all dimensions, or provide per-dimension.
        Default is 1.0. Ignored if grid is provided.
    dim : tuple of int, optional
        Spatial dimensions over which to compute derivatives. Default uses
        dimensions 1, 2, ..., n after the component dimension.
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
        Strain tensor with shape (ndim, ndim, *spatial).

    Notes
    -----
    This computes the infinitesimal (linear) strain tensor, which is
    valid for small deformations. For large deformations, consider
    the Green-Lagrange or Almansi strain tensors.

    The strain tensor is symmetric: epsilon_ij = epsilon_ji.

    Diagonal components (epsilon_ii) represent normal strains (extension/compression).
    Off-diagonal components (epsilon_ij, i != j) represent shear strains.

    The volumetric strain (dilatation) is the trace: theta = epsilon_11 + epsilon_22 + epsilon_33
    The deviatoric strain tensor is: e_ij = epsilon_ij - (1/3) * theta * delta_ij

    Examples
    --------
    >>> # Uniform extension in x-direction
    >>> n = 32
    >>> x = torch.linspace(0, 1, n)
    >>> X, Y = torch.meshgrid(x, x, indexing='ij')
    >>> displacement = torch.stack([0.1 * X, 0.0 * Y], dim=0)
    >>> eps = strain_tensor(displacement, dx=1/(n-1))
    >>> eps[0, 0].mean()  # approximately 0.1
    """
    return _strain_tensor_impl(
        displacement, dx, dim, accuracy, boundary, grid=grid
    )
