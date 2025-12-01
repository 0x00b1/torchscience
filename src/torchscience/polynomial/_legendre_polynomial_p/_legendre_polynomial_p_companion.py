import torch
from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP


def legendre_polynomial_p_companion(
    c: LegendrePolynomialP,
) -> Tensor:
    """Generate companion matrix for Legendre series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : LegendrePolynomialP
        Legendre series. Must have degree >= 1.

    Returns
    -------
    Tensor
        Companion matrix, shape (n, n) where n = degree.

    Raises
    ------
    ValueError
        If series has degree < 1.

    Notes
    -----
    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is a pure Legendre basis polynomial. This provides
    better eigenvalue estimates than the unscaled case, and for basis
    polynomials the eigenvalues are guaranteed to be real.

    The scaling is: scl[k] = 1 / sqrt(2*k + 1)

    The tridiagonal part has:
        A[k, k+1] = A[k+1, k] = (k+1) * scl[k] * scl[k+1]

    The last column is modified based on coefficients:
        A[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2*n - 1))

    Examples
    --------
    >>> c = legendre_polynomial_p(torch.tensor([0.0, 0.0, 1.0]))  # P_2
    >>> A = legendre_polynomial_p_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Special case for degree 1
    if n == 1:
        return torch.tensor(
            [[-coeffs[..., 0] / coeffs[..., 1]]],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Build companion matrix following numpy's approach
    mat = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Scaling factor: scl[k] = 1 / sqrt(2*k + 1)
    k_vals = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)
    scl = 1.0 / torch.sqrt(2 * k_vals + 1)

    # Set off-diagonal elements
    # top[k] = bot[k] = (k+1) * scl[k] * scl[k+1] for k = 0, ..., n-2
    for k in range(n - 1):
        val = (k + 1) * scl[k] * scl[k + 1]
        mat[k, k + 1] = val
        mat[k + 1, k] = val

    # Modify last column based on coefficients
    # mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2*n - 1))
    normalized_coeffs = coeffs[..., :-1] / coeffs[..., -1]
    factor = (scl / scl[-1]) * (n / (2 * n - 1))
    mat[:, -1] = mat[:, -1] - normalized_coeffs * factor

    return mat
