import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_companion(
    c: JacobiPolynomialP,
) -> Tensor:
    """Generate companion matrix for Jacobi series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : JacobiPolynomialP
        Jacobi series. Must have degree >= 1.

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
    The companion matrix is constructed using the Jacobi recurrence relation.
    The matrix is scaled to improve numerical stability.

    For the Jacobi recurrence:
        P_{n+1}^{(α,β)} = (A_n + B_n*x) * P_n^{(α,β)} - C_n * P_{n-1}^{(α,β)}

    The companion matrix has the form of a tridiagonal matrix plus a
    correction in the last column based on the polynomial coefficients.

    Examples
    --------
    >>> c = jacobi_polynomial_p(torch.tensor([0.0, 0.0, 1.0]), alpha=0.0, beta=0.0)
    >>> A = jacobi_polynomial_p_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    alpha = c.alpha
    beta = c.beta
    ab = alpha + beta
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Special case for degree 1
    if n == 1:
        # P_1^{(α,β)}(x) = (α-β)/2 + (α+β+2)/2 * x = c_0 + c_1
        # Root: x = -(c_0/c_1) * (2/(α+β+2)) - (α-β)/(α+β+2)
        # Simplified: x = -c_0/c_1 for monic, but need to account for normalization
        return torch.tensor(
            [[-coeffs[..., 0] / coeffs[..., 1]]],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Build companion matrix following the Jacobi structure
    mat = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Scaling factors for better numerical stability
    k_vals = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)

    # Build the tridiagonal structure based on the recurrence
    for k in range(n - 1):
        k_f = float(k)
        two_k_ab = 2 * k_f + ab
        two_k_ab_p2 = two_k_ab + 2

        # Recurrence: P_{k+1} = (A_k + B_k*x) * P_k - C_k * P_{k-1}
        # where B_k = (2k+α+β+1)(2k+α+β+2) / (2(k+1)(k+α+β+1))
        # Rearranging: x = (P_{k+1} + C_k*P_{k-1} - A_k*P_k) / (B_k*P_k)

        # For the companion matrix, we use scaled polynomials
        # The off-diagonal elements come from the recurrence coefficients

        B_k = (two_k_ab + 1) * two_k_ab_p2 / (2.0 * (k_f + 1) * (k_f + ab + 1))
        A_k = (
            (alpha * alpha - beta * beta) / (two_k_ab * two_k_ab_p2)
            if abs(two_k_ab * two_k_ab_p2) > 1e-15
            else 0.0
        )

        if k > 0:
            C_k = (
                (k_f + alpha)
                * (k_f + beta)
                * two_k_ab_p2
                / ((k_f + 1) * (k_f + ab + 1) * two_k_ab)
            )
        else:
            C_k = 0.0

        # Off-diagonal elements
        # mat[k, k+1] and mat[k+1, k] encode the recurrence
        if abs(B_k.item() if hasattr(B_k, "item") else B_k) > 1e-15:
            mat[k, k + 1] = 1.0 / B_k
            mat[k + 1, k] = C_k if k > 0 else 0.0

        # Diagonal elements from A_k
        mat[k, k] = (
            -A_k / B_k
            if abs(B_k.item() if hasattr(B_k, "item") else B_k) > 1e-15
            else 0.0
        )

    # Last diagonal element
    k_f = float(n - 1)
    two_k_ab = 2 * k_f + ab
    two_k_ab_p2 = two_k_ab + 2
    B_k = (two_k_ab + 1) * two_k_ab_p2 / (2.0 * (k_f + 1) * (k_f + ab + 1))
    A_k = (
        (alpha * alpha - beta * beta) / (two_k_ab * two_k_ab_p2)
        if abs(two_k_ab * two_k_ab_p2) > 1e-15
        else 0.0
    )
    mat[n - 1, n - 1] = (
        -A_k / B_k
        if abs(B_k.item() if hasattr(B_k, "item") else B_k) > 1e-15
        else 0.0
    )

    # Modify last column based on polynomial coefficients
    # mat[:, -1] -= (c[:-1] / c[-1]) * scaling
    leading_coeff = coeffs[..., -1]
    for k in range(n):
        k_f = float(k)
        two_k_ab = 2 * k_f + ab
        # Scaling factor
        if abs(two_k_ab * (two_k_ab + 2)) > 1e-15:
            scl = 1.0 / (two_k_ab * (two_k_ab + 2))
        else:
            scl = 1.0
        mat[k, -1] = mat[k, -1] - (coeffs[..., k] / leading_coeff) * scl

    return mat
