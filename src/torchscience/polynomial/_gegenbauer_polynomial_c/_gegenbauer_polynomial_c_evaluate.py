import warnings

import torch
from torch import Tensor

from torchscience.polynomial._gegenbauer_polynomial_c._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
)


def gegenbauer_polynomial_c_evaluate(
    c: GegenbauerPolynomialC,
    x: Tensor,
) -> Tensor:
    """Evaluate Gegenbauer series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : GegenbauerPolynomialC
        Gegenbauer series with coefficients shape (...batch, N) and parameter lambda_.
    x : Tensor
        Evaluation points, shape (...x_batch).

    Returns
    -------
    Tensor
        Values c(x), shape is broadcast of c's batch dims with x's shape.

    Warnings
    --------
    UserWarning
        If any evaluation points are outside the natural domain [-1, 1].

    Notes
    -----
    Uses Clenshaw's algorithm for numerical stability.

    The Gegenbauer polynomials satisfy the recurrence:
        C_0^{lambda}(x) = 1
        C_1^{lambda}(x) = 2*lambda*x
        C_{k+1}^{lambda}(x) = (2*(k+lambda)/(k+1)) * x * C_k^{lambda}(x)
                           - ((k+2*lambda-1)/(k+1)) * C_{k-1}^{lambda}(x)

    In standard form: C_{k+1}(x) = A_k * x * C_k(x) - C_k' * C_{k-1}(x)
    where A_k = 2*(k+lambda)/(k+1) and C_k' = (k+2*lambda-1)/(k+1)

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * C_k^{lambda}(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + A_k * x * b_{k+1} - C_{k+1}' * b_{k+2}  for k = n-1, ..., 1, 0
        f(x) = b_0

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> gegenbauer_polynomial_c_evaluate(c, torch.tensor([0.0]))
    tensor([-2.])  # 1 + 0 + 3*(-1) = -2 for C_2^1(0) = -1
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = GegenbauerPolynomialC.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating GegenbauerPolynomialC outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

    # Get coefficients as plain tensor
    coeffs = c.as_subclass(Tensor)
    lambda_ = c.lambda_
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

    # Ensure lambda_ is a tensor and has correct dtype
    if not isinstance(lambda_, Tensor):
        lambda_tensor = torch.tensor(lambda_, dtype=coeffs_flat.dtype)
    else:
        lambda_tensor = lambda_.clone()

    # Promote to common dtype
    common_dtype = torch.promote_types(coeffs_flat.dtype, x_flat.dtype)
    common_dtype = torch.promote_types(common_dtype, lambda_tensor.dtype)
    coeffs_flat = coeffs_flat.to(common_dtype)
    x_flat = x_flat.to(common_dtype)
    lambda_tensor = lambda_tensor.to(common_dtype)

    # Call C++ kernel: (B, N) x (M,) x () -> (B, M)
    result_flat = torch.ops.torchscience.gegenbauer_polynomial_c_evaluate(
        coeffs_flat, x_flat, lambda_tensor
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
