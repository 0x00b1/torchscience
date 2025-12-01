import torch
from torch import Tensor


def chebyshev_polynomial_u_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Chebyshev U Vandermonde matrix.

    V[i, j] = U_j(x[i])

    Parameters
    ----------
    x : Tensor
        Evaluation points, shape (n,).
    degree : int
        Maximum degree (matrix has degree+1 columns).

    Returns
    -------
    Tensor
        Vandermonde matrix, shape (n, degree+1).

    Notes
    -----
    Built using the Chebyshev U recurrence:
        U_0(x) = 1
        U_1(x) = 2x
        U_{n+1}(x) = 2*x*U_n(x) - U_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> chebyshev_polynomial_u_vandermonde(x, degree=2)
    tensor([[ 1.0000,  0.0000, -1.0000],
            [ 1.0000,  1.0000,  0.0000],
            [ 1.0000,  2.0000,  3.0000]])
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # U_0 = 1
    U_prev_prev = torch.ones_like(x)
    columns.append(U_prev_prev)

    if degree >= 1:
        # U_1 = 2x
        U_prev = 2.0 * x
        columns.append(U_prev)

        # Recurrence: U_{k+1} = 2*x*U_k - U_{k-1}
        for _ in range(2, degree + 1):
            U_curr = 2.0 * x * U_prev - U_prev_prev
            columns.append(U_curr)
            U_prev_prev = U_prev
            U_prev = U_curr

    return torch.stack(columns, dim=-1)
