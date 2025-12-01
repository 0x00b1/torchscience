import numpy as np
import torch
from torch import Tensor


def jacobi_polynomial_p_points(
    n: int,
    alpha: float,
    beta: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate Gauss-Jacobi quadrature nodes.

    These are the roots of P_n^{(α,β)}(x), optimal for polynomial integration
    and interpolation on [-1, 1] with weight w(x) = (1-x)^α * (1+x)^β.

    Parameters
    ----------
    n : int
        Number of points.
    alpha : float
        Jacobi parameter α, must be > -1.
    beta : float
        Jacobi parameter β, must be > -1.
    dtype : torch.dtype, optional
        Data type. Default is float32.
    device : torch.device or str, optional
        Device. Default is "cpu".

    Returns
    -------
    Tensor
        Gauss-Jacobi nodes in [-1, 1], sorted in descending order.

    Notes
    -----
    These points are the roots of the Jacobi polynomial P_n^{(α,β)}(x).
    Together with corresponding weights, they provide exact integration
    for polynomials up to degree 2n-1 with the Jacobi weight function.

    Examples
    --------
    >>> jacobi_polynomial_p_points(3, alpha=0.0, beta=0.0)  # Legendre case
    tensor([ 0.7746,  0.0000, -0.7746])

    >>> jacobi_polynomial_p_points(3, alpha=0.5, beta=0.5)
    tensor([...])
    """
    # Use numpy/scipy for high precision computation of roots
    # The Gauss-Jacobi quadrature nodes are the roots of P_n^{(α,β)}(x)

    # Build the companion matrix for the Jacobi polynomial P_n^{(α,β)}
    # and find its eigenvalues

    ab = alpha + beta

    # Construct the tridiagonal Jacobi matrix (symmetric)
    # whose eigenvalues are the roots of P_n^{(α,β)}
    # See Golub & Welsch (1969) for details

    if n == 0:
        return torch.tensor([], dtype=dtype, device=device)

    if n == 1:
        # Single root is at x = (α - β) / (α + β + 2)
        root = (alpha - beta) / (ab + 2)
        return torch.tensor([root], dtype=dtype, device=device)

    # Build the Jacobi matrix
    # Diagonal: a_k = (β² - α²) / ((2k + α + β)(2k + α + β + 2))
    # Off-diagonal: b_k = 2 * sqrt((k(k+α)(k+β)(k+α+β)) / ((2k+α+β-1)(2k+α+β)²(2k+α+β+1))) / (2k+α+β)

    J = np.zeros((n, n))

    for k in range(n):
        kf = float(k)
        two_k_ab = 2 * kf + ab

        # Diagonal element
        if abs(two_k_ab * (two_k_ab + 2)) > 1e-15:
            J[k, k] = (beta**2 - alpha**2) / (two_k_ab * (two_k_ab + 2))

        # Off-diagonal element (for k >= 1)
        if k > 0:
            num = kf * (kf + alpha) * (kf + beta) * (kf + ab)
            denom = (two_k_ab - 1) * two_k_ab**2 * (two_k_ab + 1)
            if denom > 1e-15:
                b_k = 2 * np.sqrt(num / denom)
                J[k, k - 1] = b_k
                J[k - 1, k] = b_k

    # Eigenvalues of J are the Jacobi quadrature nodes
    eigenvalues = np.linalg.eigvalsh(J)

    # Return in descending order
    points = np.sort(eigenvalues)[::-1]

    return torch.tensor(points.copy(), dtype=dtype, device=device)
