import torch
from torch import Tensor

from ._hermite_polynomial_h import HermitePolynomialH


def hermite_polynomial_h_companion(
    c: HermitePolynomialH,
) -> Tensor:
    """Generate companion matrix for Physicists' Hermite series.

    The eigenvalues of the companion matrix are the roots of the series.

    Parameters
    ----------
    c : HermitePolynomialH
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

    For the physicists' Hermite polynomials with recurrence:
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)

    The scaling is: scl[k] = 1 / sqrt(2^k * k!)

    Examples
    --------
    >>> c = hermite_polynomial_h(torch.tensor([0.0, 0.0, 1.0]))  # H_2
    >>> A = hermite_polynomial_h_companion(c)
    >>> A.shape
    torch.Size([2, 2])
    """
    coeffs = c.coeffs
    n = coeffs.shape[-1] - 1  # degree

    if n < 1:
        raise ValueError("Series must have degree >= 1")

    # Special case for degree 1
    if n == 1:
        # H_1(x) = 2x, so c_0 + c_1 * H_1(x) = 0 => x = -c_0 / (2*c_1)
        return torch.tensor(
            [[-coeffs[..., 0] / (2 * coeffs[..., 1])]],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )

    # Build companion matrix following numpy's approach
    mat = torch.zeros(n, n, dtype=coeffs.dtype, device=coeffs.device)

    # Scaling factor: scl[k] = 1 / sqrt(2^k * k!)
    # We compute log(scl) and exponentiate for numerical stability
    k_vals = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)

    # log(2^k * k!) = k*log(2) + log(k!)
    # log(k!) = sum_{i=1}^{k} log(i)
    log_factorial = torch.zeros(n, dtype=coeffs.dtype, device=coeffs.device)
    for k in range(1, n):
        log_factorial[k] = log_factorial[k - 1] + torch.log(
            torch.tensor(k, dtype=coeffs.dtype, device=coeffs.device)
        )

    log_scl = -0.5 * (
        k_vals
        * torch.log(
            torch.tensor(2.0, dtype=coeffs.dtype, device=coeffs.device)
        )
        + log_factorial
    )
    scl = torch.exp(log_scl)

    # Set off-diagonal elements
    # For Hermite: top[k] = bot[k] = sqrt((k+1)/2) * scl[k] / scl[k+1]
    # From recurrence H_{k+1} = 2x*H_k - 2k*H_{k-1}:
    # x = (H_{k+1} + 2k*H_{k-1}) / (2*H_k)
    # The off-diagonal is sqrt(k/2) for the symmetric tridiagonal form
    for k in range(n - 1):
        val = torch.sqrt(
            torch.tensor(
                (k + 1) / 2.0, dtype=coeffs.dtype, device=coeffs.device
            )
        )
        mat[k, k + 1] = val
        mat[k + 1, k] = val

    # Modify last column based on coefficients
    # mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * factor
    # The factor comes from the scaling relationship
    normalized_coeffs = coeffs[..., :-1] / coeffs[..., -1]
    factor = scl / scl[-1]
    # The leading coefficient of H_n is 2^n, so we need to account for this
    mat[:, -1] = mat[:, -1] - normalized_coeffs * factor / 2.0

    return mat
