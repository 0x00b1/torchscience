"""Evaluate Euler polynomial series."""

import math

import torch
from torch import Tensor

from torchscience.combinatorics._euler_number import _euler_number_exact

from ._euler_polynomial_e import EulerPolynomialE


def _euler_poly_single(n: int, x: Tensor) -> Tensor:
    """Compute single Euler polynomial E_n(x).

    Uses the formula: E_n(x) = sum_{k=0}^{n} C(n,k) * E_k / 2^k * (x - 1/2)^{n-k}
    """
    if n == 0:
        return torch.ones_like(x)

    y = x - 0.5  # Shifted variable
    result = torch.zeros_like(x)

    for k in range(n + 1):
        binom_coeff = math.comb(n, k)
        euler_k = float(_euler_number_exact(k))
        power = n - k
        coeff = binom_coeff * euler_k / (2**k)
        result = result + coeff * (y**power)

    return result


def euler_polynomial_e_evaluate(
    c: EulerPolynomialE,
    x: Tensor,
) -> Tensor:
    r"""Evaluate Euler polynomial series at points.

    Parameters
    ----------
    c : EulerPolynomialE
        Euler polynomial series with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Notes
    -----
    Evaluates f(x) = sum_{k=0}^{n} c[k] * E_k(x) where E_k(x) are Euler
    polynomials.

    The Euler polynomial E_n(x) is computed using:

    .. math::

        E_n(x) = \sum_{k=0}^{n} \binom{n}{k} \frac{E_k}{2^k} \left(x - \frac{1}{2}\right)^{n-k}

    Examples
    --------
    >>> c = euler_polynomial_e(torch.tensor([1.0, 0.0]))  # 1*E_0(x) + 0*E_1(x) = 1
    >>> euler_polynomial_e_evaluate(c, torch.tensor([0.0, 0.5, 1.0]))
    tensor([1., 1., 1.])

    >>> c = euler_polynomial_e(torch.tensor([0.0, 1.0]))  # E_1(x) = x - 1/2
    >>> euler_polynomial_e_evaluate(c, torch.tensor([0.0, 0.5, 1.0]))
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

    # Compute sum: f(x) = sum_{k=0}^{N-1} coeffs[k] * E_k(x)
    # Shape: (B, M)
    result = torch.zeros(B, M, dtype=common_dtype, device=coeffs.device)

    for k in range(N):
        # E_k(x) has shape (M,)
        E_k_x = _euler_poly_single(k, x_flat)
        # coeffs_flat[:, k] has shape (B,)
        # Outer product: (B, 1) * (1, M) -> (B, M)
        result = result + coeffs_flat[:, k : k + 1] * E_k_x.unsqueeze(0)

    # Reshape to output shape
    output_shape = batch_shape + x_shape
    return result.reshape(output_shape)
