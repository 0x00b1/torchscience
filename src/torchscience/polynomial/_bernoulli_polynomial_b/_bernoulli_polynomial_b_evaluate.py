"""Evaluate Bernoulli polynomial series."""

import math

import torch
from torch import Tensor

from torchscience.combinatorics._bernoulli_number import (
    _bernoulli_number_exact,
)

from ._bernoulli_polynomial_b import BernoulliPolynomialB


def _bernoulli_poly_single(n: int, x: Tensor) -> Tensor:
    """Compute single Bernoulli polynomial B_n(x).

    Uses the formula: B_n(x) = sum_{k=0}^{n} C(n,k) * B_k * x^{n-k}
    """
    if n == 0:
        return torch.ones_like(x)

    result = torch.zeros_like(x)
    for k in range(n + 1):
        binom_coeff = math.comb(n, k)
        bernoulli_k = float(_bernoulli_number_exact(k))
        power = n - k
        result = result + binom_coeff * bernoulli_k * (x**power)

    return result


def bernoulli_polynomial_b_evaluate(
    c: BernoulliPolynomialB,
    x: Tensor,
) -> Tensor:
    r"""Evaluate Bernoulli polynomial series at points.

    Parameters
    ----------
    c : BernoulliPolynomialB
        Bernoulli polynomial series with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Notes
    -----
    Evaluates f(x) = sum_{k=0}^{n} c[k] * B_k(x) where B_k(x) are Bernoulli
    polynomials.

    The Bernoulli polynomial B_n(x) is computed using:

    .. math::

        B_n(x) = \sum_{k=0}^{n} \binom{n}{k} B_k x^{n-k}

    Examples
    --------
    >>> c = bernoulli_polynomial_b(torch.tensor([1.0, 0.0]))  # 1*B_0(x) + 0*B_1(x) = 1
    >>> bernoulli_polynomial_b_evaluate(c, torch.tensor([0.0, 0.5, 1.0]))
    tensor([1., 1., 1.])

    >>> c = bernoulli_polynomial_b(torch.tensor([0.0, 1.0]))  # B_1(x) = x - 1/2
    >>> bernoulli_polynomial_b_evaluate(c, torch.tensor([0.0, 0.5, 1.0]))
    tensor([-0.5, 0.0, 0.5])
    """
    coeffs = c.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    # Handle trivial cases
    if n == 0:
        return x * 0.0

    batch_shape = coeffs.shape[:-1]
    x_shape = x.shape

    # Promote to common dtype
    common_dtype = torch.promote_types(coeffs.dtype, x.dtype)
    coeffs = coeffs.to(common_dtype)
    x = x.to(common_dtype)

    # Flatten for easier manipulation
    # coeffs: (...batch, N) -> (B, N)
    # x: (...x_shape) -> (M,)
    B = coeffs[..., 0].numel() if len(batch_shape) > 0 else 1
    N = coeffs.shape[-1]
    M = x.numel()

    coeffs_flat = coeffs.reshape(B, N)
    x_flat = x.reshape(M)

    # Compute sum: f(x) = sum_{k=0}^{N-1} coeffs[k] * B_k(x)
    # Shape: (B, M)
    result = torch.zeros(B, M, dtype=common_dtype, device=coeffs.device)

    for k in range(N):
        # B_k(x) has shape (M,)
        B_k_x = _bernoulli_poly_single(k, x_flat)
        # coeffs_flat[:, k] has shape (B,)
        # Outer product: (B, 1) * (1, M) -> (B, M)
        result = result + coeffs_flat[:, k : k + 1] * B_k_x.unsqueeze(0)

    # Reshape to output shape
    output_shape = batch_shape + x_shape
    return result.reshape(output_shape)
