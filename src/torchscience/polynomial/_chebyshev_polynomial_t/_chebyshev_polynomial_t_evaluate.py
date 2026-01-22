import warnings

import torch
from torch import Tensor

from torchscience.polynomial._chebyshev_polynomial_t._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
)


def chebyshev_polynomial_t_evaluate(
    c: ChebyshevPolynomialT,
    x: Tensor,
) -> Tensor:
    """Evaluate Chebyshev series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : ChebyshevPolynomialT
        Chebyshev series with coefficients shape (...batch, N).
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
    Uses Clenshaw's algorithm for numerical stability:

        b_{n+1} = b_{n+2} = 0
        b_k = c_k + 2*x*b_{k+1} - b_{k+2}  for k = n, n-1, ..., 1
        f(x) = c_0 + x*b_1 - b_2

    Examples
    --------
    >>> c = chebyshev_polynomial_t(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2*T_1 + 3*T_2
    >>> chebyshev_polynomial_t_evaluate(c, torch.tensor([0.0]))
    tensor([-2.])  # 1 + 0 + 3*(-1) = -2
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = ChebyshevPolynomialT.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating ChebyshevPolynomialT outside natural domain "
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
    result_flat = torch.ops.torchscience.chebyshev_polynomial_t_evaluate(
        coeffs_flat, x_flat
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
