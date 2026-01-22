import torch
from torch import Tensor

from torchscience.polynomial._hermite_polynomial_h._hermite_polynomial_h import (
    HermitePolynomialH,
)


def hermite_polynomial_h_evaluate(
    c: HermitePolynomialH,
    x: Tensor,
) -> Tensor:
    """Evaluate Physicists' Hermite series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : HermitePolynomialH
        Hermite series with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Notes
    -----
    Uses Clenshaw's algorithm for numerical stability.

    The physicists' Hermite polynomials satisfy the recurrence:
        H_0(x) = 1
        H_1(x) = 2x
        H_{k+1}(x) = 2x * H_k(x) - 2k * H_{k-1}(x)

    In standard form: H_{k+1}(x) = A_k * x * H_k(x) - C_k * H_{k-1}(x)
    where A_k = 2 and C_k = 2k

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * H_k(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}  for k = n-1, ..., 1, 0
        f(x) = b_0

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([1.0, 0.0, 1.0]))  # 1*H_0 + 0*H_1 + 1*H_2
    >>> hermite_polynomial_h_evaluate(c, torch.tensor([0.0]))
    tensor([-2.])  # H_0(0) = 1, H_2(0) = -2, so 1 + (-2) = -1... wait H_2(0) = 4*0^2 - 2 = -2
    """
    # No domain check for Hermite polynomials since domain is (-inf, inf)

    # The polynomial IS the coefficients tensor
    coeffs = c
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
    result_flat = torch.ops.torchscience.hermite_polynomial_h_evaluate(
        coeffs_flat, x_flat
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
