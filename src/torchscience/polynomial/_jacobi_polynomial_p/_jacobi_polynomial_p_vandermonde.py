import torch
from torch import Tensor


def jacobi_polynomial_p_vandermonde(
    x: Tensor,
    degree: int,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> Tensor:
    """Generate Jacobi Vandermonde matrix.

    V[i, j] = P_j^{(α,β)}(x[i])

    Parameters
    ----------
    x : Tensor
        Evaluation points, shape (n,).
    degree : int
        Maximum degree (matrix has degree+1 columns).
    alpha : Tensor or float
        Jacobi parameter α, must be > -1.
    beta : Tensor or float
        Jacobi parameter β, must be > -1.

    Returns
    -------
    Tensor
        Vandermonde matrix, shape (n, degree+1).

    Notes
    -----
    Built using the Jacobi recurrence:
        P_0^{(α,β)}(x) = 1
        P_1^{(α,β)}(x) = (α - β)/2 + (α + β + 2)/2 * x

        For n >= 1:
        a_n = 2(n+1)(n+α+β+1)(2n+α+β)
        b_n = (2n+α+β+1)(α²-β²)
        c_n = (2n+α+β)(2n+α+β+1)(2n+α+β+2)
        d_n = 2(n+α)(n+β)(2n+α+β+2)

        P_{n+1}^{(α,β)}(x) = ((b_n + c_n*x) * P_n - d_n * P_{n-1}) / a_n

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> jacobi_polynomial_p_vandermonde(x, degree=2, alpha=0.0, beta=0.0)
    tensor([[ 1.0000,  0.0000, -0.5000],
            [ 1.0000,  0.5000, -0.1250],
            [ 1.0000,  1.0000,  1.0000]])
    """
    # Convert alpha and beta to tensors if needed
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=x.dtype, device=x.device)

    ab = alpha + beta

    # Build columns using the recurrence
    columns = []

    # P_0^{(α,β)}(x) = 1
    P_prev_prev = torch.ones_like(x)
    columns.append(P_prev_prev)

    if degree >= 1:
        # P_1^{(α,β)}(x) = (α - β)/2 + (α + β + 2)/2 * x
        P_prev = (alpha - beta) / 2 + (ab + 2) / 2 * x
        columns.append(P_prev)

        # Recurrence for n >= 1
        for n in range(1, degree):
            n_f = float(n)
            two_n_ab = 2 * n_f + ab

            # Recurrence coefficients
            a_n = 2 * (n_f + 1) * (n_f + ab + 1) * two_n_ab
            b_n = (two_n_ab + 1) * (alpha * alpha - beta * beta)
            c_n = two_n_ab * (two_n_ab + 1) * (two_n_ab + 2)
            d_n = 2 * (n_f + alpha) * (n_f + beta) * (two_n_ab + 2)

            # P_{n+1} = ((b_n + c_n*x) * P_n - d_n * P_{n-1}) / a_n
            P_curr = ((b_n + c_n * x) * P_prev - d_n * P_prev_prev) / a_n

            columns.append(P_curr)
            P_prev_prev = P_prev
            P_prev = P_curr

    return torch.stack(columns, dim=-1)
