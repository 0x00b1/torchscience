import torch
from torch import Tensor


def chebyshev_polynomial_v_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Chebyshev V Vandermonde matrix.

    V[i, j] = V_j(x[i])

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
    Built using the Chebyshev V recurrence:
        V_0(x) = 1
        V_1(x) = 2x - 1
        V_{n+1}(x) = 2*x*V_n(x) - V_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> chebyshev_polynomial_v_vandermonde(x, degree=2)
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # V_0 = 1
    V_prev_prev = torch.ones_like(x)
    columns.append(V_prev_prev)

    if degree >= 1:
        # V_1 = 2x - 1
        V_prev = 2.0 * x - 1.0
        columns.append(V_prev)

        # Recurrence: V_{k+1} = 2*x*V_k - V_{k-1}
        for _ in range(2, degree + 1):
            V_curr = 2.0 * x * V_prev - V_prev_prev
            columns.append(V_curr)
            V_prev_prev = V_prev
            V_prev = V_curr

    return torch.stack(columns, dim=-1)
