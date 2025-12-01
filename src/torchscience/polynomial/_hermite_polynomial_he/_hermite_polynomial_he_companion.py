import torch
from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe


def hermite_polynomial_he_companion(
    c: HermitePolynomialHe,
) -> Tensor:
    """Generate companion matrix for Probabilists' Hermite series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : HermitePolynomialHe
        Hermite series. Must have degree >= 1.

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
    The companion matrix is constructed so that its eigenvalues are the
    roots of the Hermite series. The construction follows NumPy's approach
    using scaled basis polynomials for numerical stability.

    For the probabilists' Hermite polynomials with recurrence:
        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

    The scaling is: scl[k] = 1 / sqrt(k!)

    Examples
    --------
    >>> c = hermite_polynomial_he(torch.tensor([0.0, 0.0, 1.0]))  # He_2
    >>> A = hermite_polynomial_he_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Special case for degree 1
    if n == 1:
        # He_1(x) = x, so c_0 + c_1 * He_1(x) = 0 => x = -c_0 / c_1
        return torch.tensor(
            [[-coeffs[..., 0] / coeffs[..., 1]]],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Build companion matrix following numpy's approach
    mat = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Scaling factor: scl[k] = 1 / sqrt(k!)
    # We compute log(scl) and exponentiate for numerical stability
    # log(k!) = sum_{i=1}^{k} log(i)
    log_factorial = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)
    for k in range(1, n):
        log_factorial[k] = log_factorial[k - 1] + torch.log(
            torch.tensor(k, dtype=coeffs.dtype, device=coeffs.device)
        )

    log_scl = -0.5 * log_factorial
    scl = torch.exp(log_scl)

    # Set off-diagonal elements
    # For Hermite_e: from recurrence He_{k+1} = x*He_k - k*He_{k-1}:
    # x = He_{k+1} + k*He_{k-1}) / He_k
    # The off-diagonal is sqrt(k+1) for the symmetric tridiagonal form
    for k in range(n - 1):
        val = torch.sqrt(
            torch.tensor(k + 1.0, dtype=coeffs.dtype, device=coeffs.device)
        )
        mat[k, k + 1] = val
        mat[k + 1, k] = val

    # Modify last column based on coefficients
    # mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1])
    normalized_coeffs = coeffs[..., :-1] / coeffs[..., -1]
    factor = scl / scl[-1]
    mat[:, -1] = mat[:, -1] - normalized_coeffs * factor

    return mat
