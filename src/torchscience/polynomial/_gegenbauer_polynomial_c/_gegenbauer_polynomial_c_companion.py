import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_companion(
    c: GegenbauerPolynomialC,
) -> Tensor:
    """Generate companion matrix for Gegenbauer series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : GegenbauerPolynomialC
        Gegenbauer series. Must have degree >= 1.

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
    The companion matrix is derived from the three-term recurrence relation.
    For Gegenbauer polynomials with recurrence:
        C_{n+1}^{lambda}(x) = A_n * x * C_n^{lambda}(x) - B_n * C_{n-1}^{lambda}(x)

    where A_n = 2*(n+lambda)/(n+1) and B_n = (n+2*lambda-1)/(n+1)

    The companion matrix is scaled for better numerical stability.

    Examples
    --------
    >>> c = gegenbauer_polynomial_c(
    ...     torch.tensor([0.0, 0.0, 1.0]), torch.tensor(1.0)
    ... )  # C_2^1
    >>> A = gegenbauer_polynomial_c_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    lambda_ = c.lambda_
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Get lambda as scalar
    lambda_val = lambda_.item() if lambda_.dim() == 0 else lambda_[0].item()

    # Special case for degree 1
    # C_1^{lambda}(x) = 2*lambda*x, so if c_0 + c_1*C_1 = 0
    # x = -c_0 / (c_1 * 2 * lambda)
    if n == 1:
        return torch.tensor(
            [[-coeffs[..., 0] / (coeffs[..., 1] * 2.0 * lambda_val)]],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Build companion matrix using the scaled approach
    # The recurrence is: C_{k+1} = A_k * x * C_k - B_k * C_{k-1}
    # where A_k = 2*(k+lambda)/(k+1), B_k = (k+2*lambda-1)/(k+1)

    mat = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Build scaling factors for stability
    # scl[k] = 1 / norm(C_k) where norm comes from orthogonality
    k_vals = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)

    # For numerical stability, we use a symmetric scaling
    # The norm squared of C_k^{lambda} is:
    # h_k = pi * 2^{1-2*lambda} * Gamma(k+2*lambda) / ((k+lambda) * k! * Gamma(lambda)^2)
    # For simplicity, use scl[k] = 1 / sqrt(k+1)
    scl = 1.0 / torch.sqrt(k_vals + 1.0)

    # Set off-diagonal elements
    # The recurrence in matrix form becomes:
    # x * C_k = (1/A_k) * C_{k+1} + (B_k/A_k) * C_{k-1}  (for k >= 1)
    # x * C_0 = (1/A_0) * C_1

    for k in range(n - 1):
        # A_k = 2*(k+lambda)/(k+1)
        A_k = 2.0 * (k + lambda_val) / (k + 1)
        # B_{k+1} = ((k+1)+2*lambda-1)/(k+2) = (k+2*lambda)/(k+2)
        B_kp1 = (k + 2.0 * lambda_val) / (k + 2)

        # Off-diagonal: represents x * C_k = ... + (1/A_k) * C_{k+1}
        # With scaling: mat[k, k+1] = (1/A_k) * scl[k] / scl[k+1]
        val = (1.0 / A_k) * scl[k] / scl[k + 1]
        mat[k, k + 1] = val
        mat[k + 1, k] = val * (B_kp1 * A_k)  # Adjust for recurrence

    # Adjust for the polynomial coefficients
    # The companion matrix's last row/column is modified by the coefficients
    # Normalize by leading coefficient
    lead = coeffs[..., -1]
    normalized = coeffs[..., :-1] / lead

    # Modify last column
    A_nm1 = 2.0 * (n - 1 + lambda_val) / n
    for k in range(n):
        factor = (scl[k] / scl[-1]) / A_nm1
        mat[k, -1] = mat[k, -1] - normalized[..., k] * factor

    return mat
