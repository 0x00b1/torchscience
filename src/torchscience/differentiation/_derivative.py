from __future__ import annotations

import torch
from torch import Tensor
from torch.amp import custom_fwd

from torchscience.differentiation._apply import apply_stencil
from torchscience.differentiation._finite_difference_stencil import (
    finite_difference_stencil,
)


def _derivative_impl(
    field: Tensor,
    dim: int,
    order: int = 1,
    dx: float = 1.0,
    accuracy: int = 2,
    kind: str = "central",
    boundary: str = "replicate",
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
        Grid spacing. Default is 1.0.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    kind : str, optional
        Stencil type: "central", "forward", or "backward". Default is "central".
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

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
        Grid spacing. Default is 1.0.
    accuracy : int, optional
        Accuracy order of the finite difference approximation. Default is 2.
    kind : str, optional
        Stencil type: "central", "forward", or "backward". Default is "central".
    boundary : str, optional
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid".
        Default is "replicate".

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
    return _derivative_impl(field, dim, order, dx, accuracy, kind, boundary)
