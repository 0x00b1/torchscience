"""Collocation points for spectral methods.

This module provides various sets of collocation points commonly used
in spectral methods for differential equations and interpolation.
"""

import math
from typing import Optional

import torch
from torch import Tensor


def legendre_gauss_lobatto_points(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute Legendre-Gauss-Lobatto points.

    Returns n+1 points which include the endpoints -1 and 1, with the
    interior points being roots of P'_n(x).

    Parameters
    ----------
    n : int
        Number of interior points + 1. Total points = n + 1.
    dtype : torch.dtype, optional
        Data type. Defaults to float64.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Legendre-Gauss-Lobatto points in ascending order from -1 to 1.
        Shape: (n+1,).

    Notes
    -----
    These points are optimal for Legendre spectral methods and are the
    roots of (1 - x^2) * P'_n(x), where P_n is the Legendre polynomial.

    The points are computed using Newton's method with Chebyshev points
    as initial guesses.
    """
    if dtype is None:
        dtype = torch.float64

    if n == 0:
        return torch.zeros(1, dtype=dtype, device=device)

    if n == 1:
        return torch.tensor([-1.0, 1.0], dtype=dtype, device=device)

    # Start with Chebyshev-Gauss-Lobatto points as initial guess
    j = torch.arange(n + 1, dtype=dtype, device=device)
    x = -torch.cos(math.pi * j / n)

    # Newton iteration for interior points
    # The interior points are roots of P'_n(x)
    # Use the recurrence relation to compute P_n and P'_n
    max_iter = 10
    tol = 1e-15

    for _ in range(max_iter):
        # Compute P_n and P'_n using recurrence
        P_prev = torch.ones_like(x)  # P_0 = 1
        P_curr = x.clone()  # P_1 = x

        for k in range(2, n + 1):
            P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev = P_curr
            P_curr = P_next

        # P'_n = n * (x * P_n - P_{n-1}) / (x^2 - 1)
        # For interior points only
        interior_mask = (x > -1 + tol) & (x < 1 - tol)

        # Compute derivative for interior points
        dP = torch.zeros_like(x)
        x_int = x[interior_mask]
        P_curr_int = P_curr[interior_mask]
        P_prev_int = P_prev[interior_mask]

        dP[interior_mask] = (
            n * (x_int * P_curr_int - P_prev_int) / (x_int**2 - 1)
        )

        # Second derivative for Newton step
        # P''_n can be computed from P_n and P'_n
        ddP = torch.zeros_like(x)
        dP_int = dP[interior_mask]
        ddP[interior_mask] = (
            2 * x_int * dP_int - n * (n + 1) * P_curr_int
        ) / (1 - x_int**2)

        # Newton update for interior points
        delta = torch.zeros_like(x)
        nonzero_ddP = torch.abs(ddP) > tol
        delta[nonzero_ddP] = dP[nonzero_ddP] / ddP[nonzero_ddP]
        x = x - delta

        if torch.max(torch.abs(delta[interior_mask])) < tol:
            break

    # Ensure endpoints are exact
    x[0] = -1.0
    x[n] = 1.0

    return x


def uniform_points(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute uniformly spaced points on [-1, 1].

    Parameters
    ----------
    n : int
        Number of intervals (n+1 points total).
    dtype : torch.dtype, optional
        Data type. Defaults to float64.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Uniformly spaced points from -1 to 1.
        Shape: (n+1,).

    Notes
    -----
    Uniform points suffer from Runge's phenomenon for polynomial
    interpolation at high degree. Use Chebyshev or Legendre points
    for better conditioning.
    """
    if dtype is None:
        dtype = torch.float64

    return torch.linspace(-1.0, 1.0, n + 1, dtype=dtype, device=device)


def legendre_gauss_points(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute Legendre-Gauss quadrature points.

    Returns n points which are the roots of P_n(x).

    Parameters
    ----------
    n : int
        Number of points (roots of P_n).
    dtype : torch.dtype, optional
        Data type. Defaults to float64.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    Tensor
        Legendre-Gauss points in ascending order.
        Shape: (n,).

    Notes
    -----
    These points are optimal for Gaussian quadrature but do not include
    the endpoints. For methods requiring boundary conditions, use
    Legendre-Gauss-Lobatto points instead.
    """
    if dtype is None:
        dtype = torch.float64

    if n == 0:
        return torch.empty(0, dtype=dtype, device=device)

    if n == 1:
        return torch.zeros(1, dtype=dtype, device=device)

    # Use Golub-Welsch algorithm (eigenvalues of symmetric tridiagonal matrix)
    k = torch.arange(1, n, dtype=dtype, device=device)
    beta = k / torch.sqrt(4 * k**2 - 1)

    # Construct symmetric tridiagonal matrix
    T = torch.diag(beta, 1) + torch.diag(beta, -1)

    # Eigenvalues are the Gauss points
    eigenvalues, _ = torch.linalg.eigh(T)

    return eigenvalues
