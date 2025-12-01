import math

import torch
from torch import Tensor

from ._chebyshev_polynomial_v import ChebyshevPolynomialV


def chebyshev_polynomial_v_companion(
    c: ChebyshevPolynomialV,
) -> Tensor:
    """Generate companion matrix for Chebyshev V series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : ChebyshevPolynomialV
        Chebyshev V series. Must be monic (leading coefficient 1) or
        will be normalized.

    Returns
    -------
    Tensor
        Companion matrix, shape (n, n) where n = degree.

    Notes
    -----
    The Chebyshev V companion matrix is constructed similarly to other
    Chebyshev companion matrices, as a symmetric tridiagonal matrix
    that is similar to the Jacobi matrix for Chebyshev V polynomials.

    Examples
    --------
    >>> c = chebyshev_polynomial_v(torch.tensor([0.0, 0.0, 1.0]))  # V_2
    >>> A = chebyshev_polynomial_v_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Normalize to monic
    scl = coeffs[..., -1]
    coeffs = coeffs / scl.unsqueeze(-1)

    # Build companion matrix
    # The Chebyshev V companion matrix is symmetric tridiagonal
    A = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Off-diagonal values similar to other Chebyshev types
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    # Subdiagonal and superdiagonal
    for i in range(n - 1):
        if i == 0:
            A[0, 1] = sqrt2_inv
            A[1, 0] = sqrt2_inv
        else:
            A[i, i + 1] = 0.5
            A[i + 1, i] = 0.5

    # Last row modification based on coefficients
    for k in range(n):
        if k == 0:
            A[n - 1, 0] = A[n - 1, 0] - coeffs[..., 0] * sqrt2_inv
        elif k == n - 1:
            A[n - 1, k] = A[n - 1, k] - coeffs[..., k] * 0.5
        else:
            A[n - 1, k] = A[n - 1, k] - coeffs[..., k] * 0.5

    return A
