import warnings

import torch
from torch import Tensor

from ._jacobi_polynomial_p import (
    JacobiPolynomialP,
)


def jacobi_polynomial_p_evaluate(
    c: JacobiPolynomialP,
    x: Tensor,
) -> Tensor:
    """Evaluate Jacobi series at points using forward recurrence.

    Parameters
    ----------
    c : JacobiPolynomialP
        Jacobi series with coefficients shape (...batch, N) and parameters alpha, beta.
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
    Uses forward recurrence to compute the sum f(x) = sum_{k=0}^{n-1} c_k * P_k^{(alpha,beta)}(x).

    The Jacobi polynomials satisfy the three-term recurrence:
        P_0^{(alpha,beta)}(x) = 1
        P_1^{(alpha,beta)}(x) = (alpha - beta)/2 + (alpha + beta + 2)/2 * x

        For n >= 1:
        P_{n+1}^{(alpha,beta)}(x) = ((b_n + c_n*x) * P_n^{(alpha,beta)}(x) - d_n * P_{n-1}^{(alpha,beta)}(x)) / a_n

        where:
        a_n = 2(n+1)(n+alpha+beta+1)(2n+alpha+beta)
        b_n = (2n+alpha+beta+1)(alpha^2-beta^2)
        c_n = (2n+alpha+beta)(2n+alpha+beta+1)(2n+alpha+beta+2)
        d_n = 2(n+alpha)(n+beta)(2n+alpha+beta+2)

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([1.0, 2.0]), alpha=0.5, beta=0.5)
    >>> jacobi_polynomial_p_evaluate(c, torch.tensor([0.0]))
    tensor([1.])
    """
    # Domain check only applies to real tensors (complex roots are expected)
    if not x.is_complex():
        domain = JacobiPolynomialP.DOMAIN

        if ((x < domain[0]) | (x > domain[1])).any():
            warnings.warn(
                f"Evaluating JacobiPolynomialP outside natural domain "
                f"[{domain[0]}, {domain[1]}]. Results may be numerically unstable.",
                stacklevel=2,
            )

    # Get coefficients as plain tensor
    coeffs = c.as_subclass(torch.Tensor)
    alpha = c.alpha
    beta = c.beta
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

    # Ensure alpha and beta are tensors with correct dtype
    if not isinstance(alpha, Tensor):
        alpha_tensor = torch.tensor(alpha, dtype=coeffs_flat.dtype)
    else:
        alpha_tensor = alpha.clone()

    if not isinstance(beta, Tensor):
        beta_tensor = torch.tensor(beta, dtype=coeffs_flat.dtype)
    else:
        beta_tensor = beta.clone()

    # Promote to common dtype
    common_dtype = torch.promote_types(coeffs_flat.dtype, x_flat.dtype)
    common_dtype = torch.promote_types(common_dtype, alpha_tensor.dtype)
    common_dtype = torch.promote_types(common_dtype, beta_tensor.dtype)
    coeffs_flat = coeffs_flat.to(common_dtype)
    x_flat = x_flat.to(common_dtype)
    alpha_tensor = alpha_tensor.to(common_dtype)
    beta_tensor = beta_tensor.to(common_dtype)

    # Call C++ kernel: (B, N) x (M,) x () x () -> (B, M)
    result_flat = torch.ops.torchscience.jacobi_polynomial_p_evaluate(
        coeffs_flat, x_flat, alpha_tensor, beta_tensor
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
