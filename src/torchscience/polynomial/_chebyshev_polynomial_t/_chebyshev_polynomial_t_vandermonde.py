import torch
from torch import Tensor


def chebyshev_polynomial_t_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Chebyshev Vandermonde matrix.

    V[i, j] = T_j(x[i])

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
    Built using the Chebyshev recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> chebyshev_polynomial_t_vandermonde(x, degree=2)
    tensor([[ 1.0000,  0.0000, -1.0000],
            [ 1.0000,  0.5000, -0.5000],
            [ 1.0000,  1.0000,  1.0000]])
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # T_0 = 1
    T_prev_prev = torch.ones_like(x)
    columns.append(T_prev_prev)

    if degree >= 1:
        # T_1 = x
        T_prev = x
        columns.append(T_prev)

        # Recurrence: T_{k+1} = 2*x*T_k - T_{k-1}
        for _ in range(2, degree + 1):
            T_curr = 2.0 * x * T_prev - T_prev_prev
            columns.append(T_curr)
            T_prev_prev = T_prev
            T_prev = T_curr

    return torch.stack(columns, dim=-1)
