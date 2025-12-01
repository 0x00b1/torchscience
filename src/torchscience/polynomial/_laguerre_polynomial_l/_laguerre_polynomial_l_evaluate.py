import warnings

import torch
from torch import Tensor

from torchscience.polynomial._laguerre_polynomial_l._laguerre_polynomial_l import (
    LaguerrePolynomialL,
)


def laguerre_polynomial_l_evaluate(
    c: LaguerrePolynomialL,
    x: Tensor,
) -> Tensor:
    """Evaluate Laguerre series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : LaguerrePolynomialL
        Laguerre series with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Warnings
    --------
    UserWarning
        If any evaluation points are outside the natural domain [0, ∞).

    Notes
    -----
    Uses Clenshaw's algorithm for numerical stability.

    The Laguerre polynomials satisfy the recurrence:
        L_0(x) = 1
        L_1(x) = 1 - x
        L_{k+1}(x) = ((2k+1-x) * L_k(x) - k * L_{k-1}(x)) / (k+1)

    In standard form: L_{k+1}(x) = (A_k + B_k * x) * L_k(x) - C_k * L_{k-1}(x)
    where A_k = (2k+1)/(k+1), B_k = -1/(k+1), C_k = k/(k+1)

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * L_k(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + (A_k + B_k * x) * b_{k+1} - C_{k+1} * b_{k+2}  for k = n-1, ..., 0
        f(x) = b_0

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([1.0, 2.0, 3.0]))  # 1*L_0 + 2*L_1 + 3*L_2
    >>> laguerre_polynomial_l_evaluate(c, torch.tensor([0.0]))
    tensor([6.])  # 1 + 2*1 + 3*1 = 6 (since L_k(0) = 1 for all k)
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = LaguerrePolynomialL.DOMAIN

        if (x < domain[0]).any():
            warnings.warn(
                f"Evaluating LaguerrePolynomialL outside natural domain "
                f"[{domain[0]}, {domain[1]}). Results may be numerically unstable.",
                stacklevel=2,
            )

    coeffs = c.coeffs
    n = coeffs.shape[-1]

    # Handle trivial cases
    if n == 0:
        return x * 0.0

    batch_shape = coeffs.shape[:-1]
    x_shape = x.shape
    N = coeffs.shape[-1]

    # Flatten batch dimensions: (...batch, N) -> (B, N)
    B = coeffs[..., 0].numel() if len(batch_shape) > 0 else 1
    M = x.numel()

    # Clone to avoid issues with in-place modifications after evaluate
    coeffs_flat = coeffs.reshape(B, N).contiguous().clone()
    x_flat = x.reshape(M).contiguous().clone()

    # Promote to common dtype
    common_dtype = torch.promote_types(coeffs_flat.dtype, x_flat.dtype)
    coeffs_flat = coeffs_flat.to(common_dtype)
    x_flat = x_flat.to(common_dtype)

    # Call C++ kernel: (B, N) x (M,) -> (B, M)
    result_flat = torch.ops.torchscience.laguerre_polynomial_l_evaluate(
        coeffs_flat, x_flat
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
