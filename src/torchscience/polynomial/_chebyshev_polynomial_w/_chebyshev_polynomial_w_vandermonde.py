import torch
from torch import Tensor


def chebyshev_polynomial_w_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Chebyshev W Vandermonde matrix.

    V[i, j] = W_j(x[i])

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
    Built using the Chebyshev W recurrence:
        W_0(x) = 1
        W_1(x) = 2x + 1
        W_{n+1}(x) = 2*x*W_n(x) - W_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> chebyshev_polynomial_w_vandermonde(x, degree=2)
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # W_0 = 1
    W_prev_prev = torch.ones_like(x)
    columns.append(W_prev_prev)

    if degree >= 1:
        # W_1 = 2x + 1
        W_prev = 2.0 * x + 1.0
        columns.append(W_prev)

        # Recurrence: W_{k+1} = 2*x*W_k - W_{k-1}
        for _ in range(2, degree + 1):
            W_curr = 2.0 * x * W_prev - W_prev_prev
            columns.append(W_curr)
            W_prev_prev = W_prev
            W_prev = W_curr

    return torch.stack(columns, dim=-1)
