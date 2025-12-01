import torch
from torch import Tensor


def laguerre_polynomial_l_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Laguerre Vandermonde matrix.

    V[i, j] = L_j(x[i])

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
    Built using the Laguerre recurrence:
        L_0(x) = 1
        L_1(x) = 1 - x
        (n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.0])
    >>> laguerre_polynomial_l_vandermonde(x, degree=2)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.0000,  0.0000, -0.5000],
            [ 1.0000, -1.0000, -1.0000]])
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # L_0 = 1
    L_prev_prev = torch.ones_like(x)
    columns.append(L_prev_prev)

    if degree >= 1:
        # L_1 = 1 - x
        L_prev = 1.0 - x
        columns.append(L_prev)

        # Recurrence: (n+1)*L_{n+1} = (2n+1-x)*L_n - n*L_{n-1}
        for n in range(1, degree):
            L_curr = ((2 * n + 1 - x) * L_prev - n * L_prev_prev) / (n + 1)
            columns.append(L_curr)
            L_prev_prev = L_prev
            L_prev = L_curr

    return torch.stack(columns, dim=-1)
