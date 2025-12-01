import torch
from torch import Tensor


def legendre_polynomial_p_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Legendre Vandermonde matrix.

    V[i, j] = P_j(x[i])

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
    Built using the Legendre recurrence:
        P_0(x) = 1
        P_1(x) = x
        (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> legendre_polynomial_p_vandermonde(x, degree=2)
    tensor([[ 1.0000,  0.0000, -0.5000],
            [ 1.0000,  0.5000, -0.1250],
            [ 1.0000,  1.0000,  1.0000]])
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # P_0 = 1
    P_prev_prev = torch.ones_like(x)
    columns.append(P_prev_prev)

    if degree >= 1:
        # P_1 = x
        P_prev = x.clone()
        columns.append(P_prev)

        # Recurrence: (n+1)*P_{n+1} = (2n+1)*x*P_n - n*P_{n-1}
        for n in range(1, degree):
            P_curr = ((2 * n + 1) * x * P_prev - n * P_prev_prev) / (n + 1)
            columns.append(P_curr)
            P_prev_prev = P_prev
            P_prev = P_curr

    return torch.stack(columns, dim=-1)
