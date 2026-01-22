import warnings

import torch
from torch import Tensor

from torchscience.polynomial._chebyshev_polynomial_w._chebyshev_polynomial_w import (
    ChebyshevPolynomialW,
)


def chebyshev_polynomial_w_evaluate(
    c: ChebyshevPolynomialW,
    x: Tensor,
) -> Tensor:
    """Evaluate Chebyshev W series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : ChebyshevPolynomialW
        Chebyshev W series with coefficients shape (...batch, N).
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
    Uses Clenshaw's algorithm adapted for Chebyshev W polynomials.

    The Chebyshev W polynomials satisfy the recurrence:
        W_0(x) = 1
        W_1(x) = 2x + 1
        W_{n+1}(x) = 2x * W_n(x) - W_{n-1}(x)

    The Clenshaw algorithm computes:
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + 2*x*b_{k+1} - b_{k+2}  for k = n, n-1, ..., 1
        f(x) = c_0 + (2*x + 1)*b_1 - b_2

    Examples
    --------
    >>> c = chebyshev_polynomial_w(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2*W_1 + 3*W_2
    >>> chebyshev_polynomial_w_evaluate(c, torch.tensor([0.0]))
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = ChebyshevPolynomialW.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating ChebyshevPolynomialW outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

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
    result_flat = torch.ops.torchscience.chebyshev_polynomial_w_evaluate(
        coeffs_flat, x_flat
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
