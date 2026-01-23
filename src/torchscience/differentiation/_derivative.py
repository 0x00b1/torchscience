from __future__ import annotations

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._finite_difference_stencil import (
    finite_difference_stencil,
)
from torchscience.differentiation._grid import IrregularMesh, RegularGrid


def _derivative_impl(
    field: Tensor,
    dim: int,
    order: int = 1,
    dx: float = 1.0,
    accuracy: int = 2,
    kind: str = "central",
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute derivative of a scalar field along a single dimension.

    Parameters
    ----------
    field : Tensor
        Input scalar field with arbitrary shape.
    dim : int
        Dimension along which to compute the derivative.
    order : int, optional
        Order of the derivative (1 for first, 2 for second, etc.). Default is 1.
    dx : float, optional
        Grid spacing. Default is 1.0. Ignored if grid is provided.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    kind : str, optional
        Stencil type: "central", "forward", or "backward". Default is "central".
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        dx and boundary parameters.

    Returns
    -------
    Tensor
        Derivative field with the same shape as the input (unless boundary="valid").

    Examples
    --------
    >>> x = torch.linspace(0, 1, 21)
    >>> f = x**2
    >>> df = derivative(f, dim=0, order=1, dx=0.05)  # df/dx = 2x
    >>> d2f = derivative(f, dim=0, order=2, dx=0.05)  # d^2f/dx^2 = 2
    """
    # Handle sparse tensors by densifying
    if field.is_sparse:
        field = field.to_dense()

    # If grid is provided, extract spacing and boundary from it
    if grid is not None:
        if isinstance(grid, RegularGrid):
            # Normalize dimension to positive index for grid lookup
            ndim = field.ndim
            norm_dim = dim if dim >= 0 else ndim + dim
            dx = grid.spacing[norm_dim].item()
            boundary = grid.boundary
        else:
            raise NotImplementedError(
                "IrregularMesh support for derivative not yet implemented"
            )
    # Normalize dimension to positive index
    ndim = field.ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(
            f"dim {dim} out of range for tensor with {ndim} dimensions"
        )

    # Create 1D stencil
    stencil = finite_difference_stencil(
        derivative=order,
        accuracy=accuracy,
        kind=kind,
        dtype=field.dtype,
        device=field.device,
    )

    # Move the target dimension to the end, apply stencil, then move back
    # This is needed because apply_stencil operates on trailing dimensions
    perm = list(range(ndim))
    perm.remove(dim)
    perm.append(dim)

    field_permuted = field.permute(perm)
    result_permuted = apply_stencil(
        stencil, field_permuted, dx=dx, boundary=boundary
    )

    # Inverse permutation
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    return result_permuted.permute(inv_perm)


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def derivative(
    field: Tensor,
    dim: int,
    order: int = 1,
    dx: float = 1.0,
    accuracy: int = 2,
    kind: str = "central",
    boundary: str = "replicate",
    *,
    grid: RegularGrid | IrregularMesh | None = None,
) -> Tensor:
    """Compute derivative of a scalar field along a single dimension.

    Parameters
    ----------
    field : Tensor
        Input scalar field with arbitrary shape.
    dim : int
        Dimension along which to compute the derivative.
    order : int, optional
        Order of the derivative (1 for first, 2 for second, etc.). Default is 1.
    dx : float, optional
        Grid spacing. Default is 1.0. Ignored if grid is provided.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    kind : str, optional
        Stencil type: "central", "forward", or "backward". Default is "central".
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate". Ignored if grid is provided.
    grid : RegularGrid or IrregularMesh, optional
        Grid defining spacing and boundary conditions. When provided, overrides
        dx and boundary parameters.

    Returns
    -------
    Tensor
        Derivative field with the same shape as the input (unless boundary="valid").

    Examples
    --------
    >>> x = torch.linspace(0, 1, 21)
    >>> f = x**2
    >>> df = derivative(f, dim=0, order=1, dx=0.05)  # df/dx = 2x
    >>> d2f = derivative(f, dim=0, order=2, dx=0.05)  # d^2f/dx^2 = 2
    """
    return _derivative_impl(
        field, dim, order, dx, accuracy, kind, boundary, grid=grid
    )
