"""Vandermonde matrix for Bernoulli polynomial basis."""

import math

import torch
from torch import Tensor

from torchscience.combinatorics._bernoulli_number import (
    _bernoulli_number_exact,
)


def bernoulli_polynomial_b_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Compute Vandermonde matrix for Bernoulli polynomial basis.

    The Vandermonde matrix V has entries V[i, j] = B_j(x[i]).

    Parameters
    ----------
    x : Tensor
        Sample points, shape (..., M).
    degree : int
        Maximum degree of Bernoulli polynomials.

    Returns
    -------
    Tensor
        Vandermonde matrix of shape (..., M, degree+1).

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> V = bernoulli_polynomial_b_vandermonde(x, 2)
    >>> V.shape
    torch.Size([3, 3])
    """
    x_flat = x.reshape(-1)
    M = x_flat.shape[0]
    N = degree + 1

    V = torch.zeros(M, N, dtype=x.dtype, device=x.device)

    # Compute B_n(x) for each n
    for n in range(N):
        # B_n(x) = sum_{k=0}^{n} C(n,k) * B_k * x^{n-k}
        B_n = torch.zeros_like(x_flat)
        for k in range(n + 1):
            binom = math.comb(n, k)
            bernoulli_k = float(_bernoulli_number_exact(k))
            power = n - k
            B_n = B_n + binom * bernoulli_k * (x_flat**power)
        V[:, n] = B_n

    # Reshape to match input batch shape
    output_shape = x.shape + (N,)
    return V.reshape(output_shape)
