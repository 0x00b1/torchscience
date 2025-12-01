import torch
from torch import Tensor

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_companion(
    c: LaguerrePolynomialL,
) -> Tensor:
    """Generate companion matrix for Laguerre series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : LaguerrePolynomialL
        Laguerre series. Must have degree >= 1.

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
    The companion matrix is constructed from the Laguerre recurrence:
        (k+1) L_{k+1}(x) = (2k+1-x) L_k(x) - k L_{k-1}(x)

    Rearranging for x * L_k:
        x L_k(x) = (2k+1) L_k(x) - (k+1) L_{k+1}(x) - k L_{k-1}(x)

    The matrix is scaled to be symmetric for better eigenvalue computation.
    The scaling is: scl[k] = 1 (Laguerre polynomials are already normalized).

    Examples
    --------
    >>> c = laguerre_polynomial_l(torch.tensor([0.0, 0.0, 1.0]))  # L_2
    >>> A = laguerre_polynomial_l_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Special case for degree 1: L_1(x) = 1 - x, root at x = 1
    # For c[0] + c[1] * L_1 = c[0] + c[1] * (1 - x) = 0
    # => x = (c[0] + c[1]) / c[1] = 1 + c[0]/c[1]
    if n == 1:
        return torch.tensor(
            [[1.0 + coeffs[..., 0] / coeffs[..., 1]]],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Build companion matrix following numpy's approach
    mat = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # For Laguerre, the recurrence x L_k = (2k+1) L_k - (k+1) L_{k+1} - k L_{k-1}
    # gives a symmetric tridiagonal companion matrix.
    #
    # Diagonal: mat[k,k] = 2k + 1
    # Off-diagonal: mat[k,k+1] = mat[k+1,k] = -(k+1)
    #
    # But we need to scale and adjust for the polynomial coefficients.

    # Using numpy's approach: build symmetric tridiagonal
    # diag[k] = 2*k + 1
    # off_diag[k] = k + 1

    for k in range(n):
        mat[k, k] = 2 * k + 1

    for k in range(n - 1):
        val = k + 1
        mat[k, k + 1] = -val
        mat[k + 1, k] = -val

    # Modify last column based on coefficients
    # This adjusts for the actual polynomial (not just L_n)
    normalized_coeffs = coeffs[..., :-1] / coeffs[..., -1]
    # The adjustment factor for Laguerre
    # Following the pattern from numpy's lagcompanion
    mat[:, -1] = mat[:, -1] + normalized_coeffs * (
        (-1) ** torch.arange(n, device=coeffs.device)
    )

    return mat
