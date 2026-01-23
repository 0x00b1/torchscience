"""Chebyshev spectral differentiation for non-periodic domains."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.amp import custom_fwd


def chebyshev_points(
    n: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Generate Chebyshev-Gauss-Lobatto points.

    Points are x_j = cos(pi*j/n) for j = 0, 1, ..., n.
    These are the extrema of the Chebyshev polynomial T_n(x).

    Parameters
    ----------
    n : int
        Polynomial degree (returns n+1 points).
    device : torch.device, optional
        Device for output tensor.
    dtype : torch.dtype, optional
        Data type for output tensor.

    Returns
    -------
    Tensor
        Chebyshev points of shape (n+1,), ordered from 1 to -1.

    Examples
    --------
    >>> x = chebyshev_points(4)
    >>> x
    tensor([ 1.0000,  0.7071,  0.0000, -0.7071, -1.0000])
    """
    j = torch.arange(n + 1, device=device, dtype=dtype or torch.float64)
    return torch.cos(math.pi * j / n)


def _chebyshev_diff_matrix(
    n: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Compute Chebyshev differentiation matrix.

    The matrix D satisfies: (Df)_i = f'(x_i) where x_i are Chebyshev points.

    Parameters
    ----------
    n : int
        Polynomial degree.
    device : torch.device
        Device for output.
    dtype : torch.dtype
        Data type for output.

    Returns
    -------
    Tensor
        Differentiation matrix of shape (n+1, n+1).
    """
    x = chebyshev_points(n, device=device, dtype=dtype)

    # Coefficients c_i = 2 for endpoints, 1 otherwise
    c = torch.ones(n + 1, device=device, dtype=dtype)
    c[0] = 2.0
    c[n] = 2.0

    # Alternating signs
    c = c * ((-1.0) ** torch.arange(n + 1, device=device, dtype=dtype))

    # Build differentiation matrix
    # D_ij = c_i / (c_j * (x_i - x_j)) for i != j
    # D_ii = -sum_{j != i} D_ij

    X = x.unsqueeze(1)  # (n+1, 1)
    Y = x.unsqueeze(0)  # (1, n+1)

    # x_i - x_j with small epsilon to avoid division by zero on diagonal
    dX = X - Y
    dX = dX + torch.eye(
        n + 1, device=device, dtype=dtype
    )  # Add 1 on diagonal temporarily

    C = c.unsqueeze(1) / c.unsqueeze(0)  # c_i / c_j

    D = C / dX

    # Set diagonal to zero temporarily
    D = D - torch.diag(torch.diag(D))

    # Diagonal entries: D_ii = -sum_{j != i} D_ij
    D = D - torch.diag(D.sum(dim=1))

    return D


def _chebyshev_derivative_impl(
    field: Tensor,
    dim: int,
    order: int = 1,
) -> Tensor:
    """Internal implementation of Chebyshev derivative."""
    ndim = field.ndim
    if dim < 0:
        dim = ndim + dim

    n = field.shape[dim] - 1  # Polynomial degree

    # Get differentiation matrix
    D = _chebyshev_diff_matrix(n, field.device, field.dtype)

    # Apply D along the specified dimension
    # Move dim to last position, apply D, move back
    result = field
    for _ in range(order):
        result = torch.tensordot(result, D, dims=([dim], [1]))
        # tensordot puts the result dimension at the end, need to move it back
        if dim != ndim - 1:
            result = result.movedim(-1, dim)

    return result


@custom_fwd(device_type="cpu", cast_inputs=torch.float32)
def chebyshev_derivative(
    field: Tensor,
    dim: int,
    order: int = 1,
) -> Tensor:
    """Compute derivative using Chebyshev spectral method.

    Assumes the field is sampled at Chebyshev-Gauss-Lobatto points
    along the specified dimension. Achieves spectral accuracy for
    smooth functions on non-periodic domains [-1, 1].

    Parameters
    ----------
    field : Tensor
        Input field sampled at Chebyshev points.
    dim : int
        Dimension along which to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    Tensor
        The derivative at Chebyshev points.

    Notes
    -----
    Unlike FFT-based spectral methods, Chebyshev differentiation
    works for non-periodic domains. The function should be sampled
    at x_j = cos(pi*j/N) for j = 0, ..., N.

    For functions on [a, b] instead of [-1, 1], scale the derivative
    by 2/(b-a).

    Examples
    --------
    >>> n = 32
    >>> x = chebyshev_points(n)
    >>> f = torch.sin(x)
    >>> df = chebyshev_derivative(f, dim=0)  # approximately cos(x)
    """
    return _chebyshev_derivative_impl(field, dim, order)
