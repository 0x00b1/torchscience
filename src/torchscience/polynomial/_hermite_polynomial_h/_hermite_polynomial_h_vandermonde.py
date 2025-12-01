import torch
from torch import Tensor


def hermite_polynomial_h_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Physicists' Hermite Vandermonde matrix.

    V[i, j] = H_j(x[i])

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
    Built using the Hermite recurrence:
        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> hermite_polynomial_h_vandermonde(x, degree=2)
    tensor([[ 1.0000,  0.0000, -2.0000],
            [ 1.0000,  1.0000, -1.0000],
            [ 1.0000,  2.0000,  2.0000]])
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # H_0 = 1
    H_prev_prev = torch.ones_like(x)
    columns.append(H_prev_prev)

    if degree >= 1:
        # H_1 = 2x
        H_prev = 2.0 * x
        columns.append(H_prev)

        # Recurrence: H_{n+1} = 2x * H_n - 2n * H_{n-1}
        for n in range(1, degree):
            H_curr = 2.0 * x * H_prev - 2.0 * n * H_prev_prev
            columns.append(H_curr)
            H_prev_prev = H_prev
            H_prev = H_curr

    return torch.stack(columns, dim=-1)
