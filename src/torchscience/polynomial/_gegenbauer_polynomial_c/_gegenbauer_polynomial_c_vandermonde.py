import torch
from torch import Tensor


def gegenbauer_polynomial_c_vandermonde(
    x: Tensor,
    degree: int,
    lambda_: Tensor,
) -> Tensor:
    """Generate Gegenbauer Vandermonde matrix.

    V[i, j] = C_j^{lambda}(x[i])

    Parameters
    ----------
    x : Tensor
        Evaluation points, shape (n,).
    degree : int
        Maximum degree (matrix has degree+1 columns).
    lambda_ : Tensor
        Parameter lambda > -1/2.

    Returns
    -------
    Tensor
        Vandermonde matrix, shape (n, degree+1).

    Notes
    -----
    Built using the Gegenbauer recurrence:
        C_0^{lambda}(x) = 1
        C_1^{lambda}(x) = 2*lambda*x
        C_{n+1}^{lambda}(x) = (2*(n+lambda)/(n+1))*x*C_n^{lambda}(x)
                           - ((n+2*lambda-1)/(n+1))*C_{n-1}^{lambda}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> gegenbauer_polynomial_c_vandermonde(x, degree=2, lambda_=torch.tensor(1.0))
    tensor([[ 1.0000,  0.0000, -1.0000],
            [ 1.0000,  1.0000,  0.5000],
            [ 1.0000,  2.0000,  3.0000]])
    """
    # Ensure lambda_ is a scalar
    if lambda_.dim() > 0:
        lambda_ = lambda_.squeeze()

    # Build columns without in-place ops for autograd compatibility
    columns = []

    # C_0^{lambda} = 1
    C_prev_prev = torch.ones_like(x)
    columns.append(C_prev_prev)

    if degree >= 1:
        # C_1^{lambda} = 2*lambda*x
        C_prev = 2.0 * lambda_ * x
        columns.append(C_prev)

        # Recurrence: C_{n+1} = (2*(n+lambda)/(n+1))*x*C_n - ((n+2*lambda-1)/(n+1))*C_{n-1}
        for n in range(1, degree):
            a_n = 2.0 * (n + lambda_) / (n + 1)
            c_n = (n + 2.0 * lambda_ - 1.0) / (n + 1)
            C_curr = a_n * x * C_prev - c_n * C_prev_prev
            columns.append(C_curr)
            C_prev_prev = C_prev
            C_prev = C_curr

    return torch.stack(columns, dim=-1)
