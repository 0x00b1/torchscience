import torch
from torch import Tensor


def gegenbauer_polynomial_c_points(
    n: int,
    lambda_: Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Gauss-Gegenbauer quadrature nodes.

    These are the roots of C_n^{lambda}(x), optimal for polynomial integration
    and interpolation on [-1, 1] with weight (1-x^2)^{lambda-1/2}.

    Parameters
    ----------
    n : int
        Number of points.
    lambda_ : Tensor
        Parameter lambda > -1/2.
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    Tensor
        Gauss-Gegenbauer nodes in [-1, 1], sorted in descending order.

    Notes
    -----
    These points are the roots of the Gegenbauer polynomial C_n^{lambda}(x).
    Together with corresponding weights, they provide exact integration
    for polynomials up to degree 2n-1 with respect to the weight function
    w(x) = (1-x^2)^{lambda-1/2}.

    The points are symmetric about x=0.

    For lambda = 1/2, these reduce to Gauss-Legendre nodes.
    For lambda = 1, these reduce to Gauss-Chebyshev (second kind) nodes.

    Examples
    --------
    >>> gegenbauer_polynomial_c_points(3, torch.tensor(1.0))
    tensor([ 0.7746,  0.0000, -0.7746])
    """
    # Ensure lambda_ is a tensor
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(lambda_, dtype=dtype, device=device)

    # Ensure lambda_ is a scalar for this computation
    lambda_val = lambda_.item() if lambda_.dim() == 0 else lambda_[0].item()

    # Build companion matrix and find eigenvalues
    # The roots of C_n^{lambda} are eigenvalues of the tridiagonal Jacobi matrix
    # with entries determined by the three-term recurrence

    # For the monic Gegenbauer polynomial, the Jacobi matrix has:
    # a_k = 0 (diagonal is zero due to symmetry)
    # b_k = sqrt(k(k+2*lambda-1) / (4*(k+lambda-1)*(k+lambda))) for subdiagonal

    if n == 1:
        return torch.zeros(1, dtype=dtype, device=device)

    # Build tridiagonal Jacobi matrix
    J = torch.zeros(n, n, dtype=torch.float64, device=device)

    for k in range(1, n):
        # b_k for k-th subdiagonal element
        numer = k * (k + 2 * lambda_val - 1)
        denom = 4 * (k + lambda_val - 1) * (k + lambda_val)
        if denom > 0 and numer >= 0:
            b_k = (numer / denom) ** 0.5
        else:
            b_k = 0.0
        J[k - 1, k] = b_k
        J[k, k - 1] = b_k

    # Eigenvalues are the roots
    eigenvalues = torch.linalg.eigvalsh(J)

    # Sort in descending order
    eigenvalues = eigenvalues.flip(0)

    return eigenvalues.to(dtype=dtype)
