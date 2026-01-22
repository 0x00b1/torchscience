import torch
from torch import Tensor

from ._chebyshev_polynomial_u import ChebyshevPolynomialU


def chebyshev_polynomial_u_companion(
    c: ChebyshevPolynomialU,
) -> Tensor:
    """Generate companion matrix for Chebyshev U series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : ChebyshevPolynomialU
        Chebyshev U series. Must be monic (leading coefficient 1) or
        will be normalized.

    Returns
    -------
    Tensor
        Companion matrix, shape (n, n) where n = degree.

    Notes
    -----
    The Chebyshev U companion matrix is a symmetric tridiagonal matrix.
    For U polynomials, the recurrence is U_{n+1} = 2x*U_n - U_{n-1},
    which leads to off-diagonal elements of 0.5.

    Examples
    --------
    >>> c = chebyshev_polynomial_u(torch.tensor([0.0, 0.0, 1.0]))  # U_2
    >>> A = chebyshev_polynomial_u_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.as_subclass(torch.Tensor)
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Normalize to monic (leading coefficient = 1)
    scl = coeffs[..., -1]
    coeffs = coeffs / scl.unsqueeze(-1)

    # Build companion matrix
    # For Chebyshev U, the companion matrix is symmetric tridiagonal
    # with off-diagonal elements 0.5 (from the recurrence 2x*U_n = U_{n+1} + U_{n-1})
    A = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Off-diagonal values: 0.5
    for i in range(n - 1):
        A[i, i + 1] = 0.5
        A[i + 1, i] = 0.5

    # Last row modification based on coefficients
    # The last row encodes: -c_0/c_n, -c_1/c_n, ..., -c_{n-1}/c_n
    # but scaled by the recurrence factor
    for k in range(n):
        A[n - 1, k] = A[n - 1, k] - coeffs[..., k] * 0.5

    return A
