import warnings

import torch
from torch import Tensor

from torchscience.polynomial._legendre_polynomial_p._legendre_polynomial_p import (
    LegendrePolynomialP,
)


def legendre_polynomial_p_evaluate(
    c: LegendrePolynomialP,
    x: Tensor,
) -> Tensor:
    """Evaluate Legendre series at points using Clenshaw's algorithm.

    Parameters
    ----------
    c : LegendrePolynomialP
        Legendre series with coefficients shape (...batch, N).
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

    The Legendre polynomials satisfy the recurrence:
        P_0(x) = 1
        P_1(x) = x
        P_{k+1}(x) = ((2k+1)/(k+1)) * x * P_k(x) - (k/(k+1)) * P_{k-1}(x)

    In standard form: P_{k+1}(x) = A_k * x * P_k(x) - C_k * P_{k-1}(x)
    where A_k = (2k+1)/(k+1) and C_k = k/(k+1)

    For the Clenshaw backward recurrence to evaluate f(x) = sum(c_k * P_k(x)):
        b_{n+1} = b_{n+2} = 0
        b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}  for k = n-1, ..., 1, 0
        f(x) = b_0

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([1.0, 2.0, 3.0]))  # 1*P_0 + 2*P_1 + 3*P_2
    >>> legendre_polynomial_p_evaluate(c, torch.tensor([0.0]))
    tensor([-0.5])  # 1 + 0 + 3*(-1/2) = -0.5
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = LegendrePolynomialP.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating LegendrePolynomialP outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

    coeffs = c.as_subclass(torch.Tensor)
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
    result_flat = torch.ops.torchscience.legendre_polynomial_p_evaluate(
        coeffs_flat, x_flat
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
