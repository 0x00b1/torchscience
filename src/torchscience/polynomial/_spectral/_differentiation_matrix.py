"""General polynomial differentiation matrices.

This module provides utilities for constructing differentiation matrices
for various polynomial interpolation schemes.
"""

from typing import Optional

import torch
from torch import Tensor


def lagrange_differentiation_matrix(
    x: Tensor,
) -> Tensor:
    """Compute the Lagrange differentiation matrix for arbitrary points.

    Given a set of n+1 distinct points, returns the (n+1) x (n+1)
    differentiation matrix D such that (Df)_i ≈ f'(x_i).

    Parameters
    ----------
    x : Tensor
        Collocation points. Shape: (n+1,).

    Returns
    -------
    Tensor
        Differentiation matrix D. Shape: (n+1, n+1).

    Notes
    -----
    The Lagrange interpolation polynomial is:

        p(x) = sum_j f_j * L_j(x)

    where L_j(x) = prod_{k≠j} (x - x_k) / (x_j - x_k)

    The differentiation matrix entries are:

        D_ij = L'_j(x_i) = w_j / (w_i * (x_i - x_j)),  i ≠ j
        D_ii = -sum_{j≠i} D_ij

    where w_j = 1 / prod_{k≠j} (x_j - x_k) are the barycentric weights.

    This formula is numerically stable and works for any set of distinct
    points. For well-conditioned interpolation, use Chebyshev or
    Legendre-Gauss-Lobatto points.
    """
    n = x.shape[0] - 1
    dtype = x.dtype
    device = x.device

    # Compute barycentric weights
    # w_j = 1 / prod_{k≠j} (x_j - x_k)
    X = x.unsqueeze(1) - x.unsqueeze(0)  # x_i - x_j

    # Set diagonal to 1 to avoid log(0)
    X_offdiag = X.clone()
    X_offdiag.fill_diagonal_(1.0)

    # Compute weights using log for numerical stability
    log_abs_X = torch.log(torch.abs(X_offdiag))
    sign_X = torch.sign(X_offdiag)

    # Sum of log|x_j - x_k| for k ≠ j
    log_sum = log_abs_X.sum(dim=1)
    sign_prod = sign_X.prod(dim=1)

    # w_j = sign * exp(-log_sum)
    w = sign_prod * torch.exp(-log_sum)

    # Build differentiation matrix
    # D_ij = w_j / (w_i * (x_i - x_j)) for i ≠ j
    W = w.unsqueeze(0) / w.unsqueeze(1)  # w_j / w_i
    D = W / X

    # Set diagonal to zero before computing sum
    D.fill_diagonal_(0.0)

    # Diagonal: D_ii = -sum_{j≠i} D_ij
    D.diagonal().copy_(-D.sum(dim=1))

    return D


def integration_matrix(
    x: Tensor,
) -> Tensor:
    """Compute the spectral integration matrix.

    Given collocation points x, returns a matrix S such that (Sf)_i
    approximates the integral of f from x_0 to x_i.

    Parameters
    ----------
    x : Tensor
        Collocation points. Shape: (n+1,).

    Returns
    -------
    Tensor
        Integration matrix S. Shape: (n+1, n+1).

    Notes
    -----
    The integration matrix is the pseudo-inverse of the differentiation
    matrix with the constraint that the integral at x_0 is zero:

        S_ij = int_{x_0}^{x_i} L_j(t) dt

    For Chebyshev points, this uses Clenshaw-Curtis integration weights.
    For general points, it computes the integrals of Lagrange polynomials.
    """
    n = x.shape[0] - 1
    dtype = x.dtype
    device = x.device

    # Compute Lagrange polynomial integrals analytically is complex
    # Instead, use the fact that S ≈ D^{-1} with appropriate boundary conditions

    # For small n, compute integrals of Lagrange polynomials directly
    # L_j(t) = prod_{k≠j} (t - x_k) / (x_j - x_k)
    # int L_j dt is a polynomial of degree n+1

    S = torch.zeros((n + 1, n + 1), dtype=dtype, device=device)

    # For each Lagrange polynomial L_j, compute its indefinite integral
    for j in range(n + 1):
        # Coefficients of L_j in power basis
        # L_j(t) = prod_{k≠j} (t - x_k) / (x_j - x_k)
        # Start with constant 1
        coeffs = torch.tensor([1.0], dtype=dtype, device=device)

        for k in range(n + 1):
            if k == j:
                continue
            # Multiply by (t - x_k) / (x_j - x_k)
            scale = 1.0 / (x[j] - x[k])
            # (a_0 + a_1*t + ... + a_m*t^m) * (t - x_k)
            # = -x_k*a_0 + (a_0 - x_k*a_1)*t + (a_1 - x_k*a_2)*t^2 + ...
            new_coeffs = torch.zeros(
                len(coeffs) + 1, dtype=dtype, device=device
            )
            new_coeffs[:-1] -= x[k] * coeffs
            new_coeffs[1:] += coeffs
            coeffs = new_coeffs * scale

        # Integrate: a_k*t^k -> a_k*t^{k+1}/(k+1)
        k_idx = torch.arange(len(coeffs), dtype=dtype, device=device)
        int_coeffs = coeffs / (k_idx + 1)
        int_coeffs = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), int_coeffs]
        )

        # Evaluate at each x_i - evaluate at x_0
        for i in range(n + 1):
            # Evaluate integral polynomial at x_i and x_0
            val_i = _eval_poly(int_coeffs, x[i])
            val_0 = _eval_poly(int_coeffs, x[0])
            S[i, j] = val_i - val_0

    return S


def _eval_poly(coeffs: Tensor, x: Tensor) -> Tensor:
    """Evaluate polynomial with coefficients in ascending order using Horner's method."""
    # PyTorch doesn't support negative step slicing, so we flip the tensor
    coeffs_reversed = coeffs.flip(0)
    result = coeffs_reversed[0]
    for i in range(1, len(coeffs_reversed)):
        result = result * x + coeffs_reversed[i]
    return result


def legendre_differentiation_matrix(
    n: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Compute Legendre spectral differentiation matrix.

    Returns the differentiation matrix D for the Legendre-Gauss-Lobatto
    collocation points.

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
    D : Tensor
        Differentiation matrix. Shape: (n+1, n+1).
    x : Tensor
        Legendre-Gauss-Lobatto points. Shape: (n+1,).
    """
    from ._collocation_points import legendre_gauss_lobatto_points

    if dtype is None:
        dtype = torch.float64

    x = legendre_gauss_lobatto_points(n, dtype=dtype, device=device)
    D = lagrange_differentiation_matrix(x)

    return D, x
