"""Vandermonde matrix for Euler polynomial basis."""

import math

import torch
from torch import Tensor

from torchscience.combinatorics._euler_number import _euler_number_exact


def euler_polynomial_e_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Compute Vandermonde matrix for Euler polynomial basis.

    The Vandermonde matrix V has entries V[i, j] = E_j(x[i]).

    Parameters
    ----------
    x : Tensor
        Sample points, shape (..., M).
    degree : int
        Maximum degree of Euler polynomials.

    Returns
    -------
    Tensor
        Vandermonde matrix of shape (..., M, degree+1).
    """
    x_flat = x.reshape(-1)
    M = x_flat.shape[0]
    N = degree + 1

    V = torch.zeros(M, N, dtype=x.dtype, device=x.device)

    # Compute E_n(x) for each n
    for n in range(N):
        # E_n(x) = sum_{k=0}^{n} C(n,k) * E_k / 2^k * (x - 1/2)^{n-k}
        y = x_flat - 0.5
        E_n = torch.zeros_like(x_flat)
        for k in range(n + 1):
            binom = math.comb(n, k)
            euler_k = float(_euler_number_exact(k))
            power = n - k
            coeff = binom * euler_k / (2**k)
            E_n = E_n + coeff * (y**power)
        V[:, n] = E_n

    # Reshape to match input batch shape
    output_shape = x.shape + (N,)
    return V.reshape(output_shape)
