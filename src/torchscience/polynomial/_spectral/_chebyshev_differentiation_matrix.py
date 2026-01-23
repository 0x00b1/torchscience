"""Chebyshev spectral differentiation matrices.

Chebyshev spectral methods use Chebyshev-Gauss-Lobatto points (extrema of
Chebyshev polynomials) for interpolation. The differentiation matrix maps
function values at these points to derivative values.
"""

import math
from typing import Optional

import torch
from torch import Tensor


def chebyshev_differentiation_matrix(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute the Chebyshev spectral differentiation matrix.

    Returns an (n+1) x (n+1) matrix D such that Df approximates the derivative
    of f at the Chebyshev-Gauss-Lobatto points.

    Parameters
    ----------
    n : int
        Polynomial degree. The matrix will have shape (n+1, n+1).
    dtype : torch.dtype, optional
        Data type. Defaults to float64 for numerical stability.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Differentiation matrix D of shape (n+1, n+1).

    Notes
    -----
    The Chebyshev-Gauss-Lobatto points are:

        x_j = cos(pi * j / n),  j = 0, 1, ..., n

    These are the extrema of T_n(x), including the endpoints x = +/- 1.

    The differentiation matrix D satisfies:

        (Df)_i ≈ f'(x_i)

    where f is a vector of function values at the Chebyshev points.

    The matrix entries are given by:

        D_ij = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j),  i ≠ j
        D_ii = -sum_{j≠i} D_ij

    where c_0 = c_n = 2 and c_j = 1 for 0 < j < n.

    For spectral methods on the interval [a, b], scale by 2/(b-a).

    Examples
    --------
    >>> D = chebyshev_differentiation_matrix(4)
    >>> # Differentiate f(x) = x^2 at Chebyshev points
    >>> x = chebyshev_points(4)
    >>> f = x ** 2
    >>> df = D @ f  # Approximates 2*x

    References
    ----------
    .. [1] Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
    .. [2] Weideman, J. A. C., & Reddy, S. C. (2000). A MATLAB differentiation
           matrix suite. ACM Transactions on Mathematical Software.
    """
    if dtype is None:
        dtype = torch.float64

    if n == 0:
        return torch.zeros((1, 1), dtype=dtype, device=device)

    # Chebyshev-Gauss-Lobatto points
    j = torch.arange(n + 1, dtype=dtype, device=device)
    x = torch.cos(math.pi * j / n)

    # Coefficients c_j: c_0 = c_n = 2, c_j = 1 for 0 < j < n
    c = torch.ones(n + 1, dtype=dtype, device=device)
    c[0] = 2.0
    c[n] = 2.0

    # Alternating signs
    c = c * ((-1.0) ** j)

    # Build differentiation matrix
    # D_ij = (c_i / c_j) / (x_i - x_j) for i ≠ j
    X = x.unsqueeze(1) - x.unsqueeze(0)  # x_i - x_j

    # Handle diagonal separately
    X.fill_diagonal_(1.0)  # Avoid division by zero

    # Off-diagonal entries
    C = c.unsqueeze(1) / c.unsqueeze(0)  # c_i / c_j
    D = C / X

    # Set diagonal to zero before computing diagonal sum
    D.fill_diagonal_(0.0)

    # Diagonal entries: D_ii = -sum_{j≠i} D_ij
    D.diagonal().copy_(-D.sum(dim=1))

    return D


def chebyshev_points(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute Chebyshev-Gauss-Lobatto points.

    Returns n+1 points which are the extrema of T_n(x), including endpoints.

    Parameters
    ----------
    n : int
        Polynomial degree.
    dtype : torch.dtype, optional
        Data type. Defaults to float64.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Points x_j = cos(pi * j / n) for j = 0, 1, ..., n.
        Shape: (n+1,). Points are in descending order from 1 to -1.
    """
    if dtype is None:
        dtype = torch.float64

    if n == 0:
        return torch.zeros(1, dtype=dtype, device=device)

    j = torch.arange(n + 1, dtype=dtype, device=device)
    return torch.cos(math.pi * j / n)


def chebyshev_differentiation_matrix_2(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute the second-order Chebyshev differentiation matrix.

    Returns an (n+1) x (n+1) matrix D2 = D @ D where D is the first-order
    Chebyshev differentiation matrix.

    Parameters
    ----------
    n : int
        Polynomial degree.
    dtype : torch.dtype, optional
        Data type. Defaults to float64.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Second-order differentiation matrix D2 of shape (n+1, n+1).

    Notes
    -----
    For efficiency, this is computed as D @ D rather than deriving
    a specialized formula.
    """
    D = chebyshev_differentiation_matrix(n, dtype=dtype, device=device)
    return D @ D


def chebyshev_differentiation_matrix_scaled(
    n: int,
    a: float,
    b: float,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Compute Chebyshev differentiation matrix for interval [a, b].

    Returns both the differentiation matrix and the collocation points
    scaled to the interval [a, b].

    Parameters
    ----------
    n : int
        Polynomial degree.
    a : float
        Left endpoint of interval.
    b : float
        Right endpoint of interval.
    dtype : torch.dtype, optional
        Data type. Defaults to float64.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    D : Tensor
        Differentiation matrix scaled for interval [a, b]. Shape: (n+1, n+1).
    x : Tensor
        Collocation points on [a, b]. Shape: (n+1,).

    Notes
    -----
    The standard Chebyshev points are on [-1, 1]. For an arbitrary interval
    [a, b], the linear transformation is:

        y = (b - a) / 2 * x + (b + a) / 2

    The differentiation matrix is scaled by 2 / (b - a) to account for
    the chain rule.
    """
    D_std = chebyshev_differentiation_matrix(n, dtype=dtype, device=device)
    x_std = chebyshev_points(n, dtype=dtype, device=device)

    # Scale factor for differentiation
    scale = 2.0 / (b - a)
    D = scale * D_std

    # Map points from [-1, 1] to [a, b]
    x = (b - a) / 2.0 * x_std + (b + a) / 2.0

    return D, x
