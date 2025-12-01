import torch
from torch import Tensor


def hermite_polynomial_he_vandermonde(
    x: Tensor,
    degree: int,
) -> Tensor:
    """Generate Probabilists' Hermite Vandermonde matrix.

    V[i, j] = He_j(x[i])

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
    Built using the Hermite_e recurrence:
        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> hermite_polynomial_he_vandermonde(x, degree=2)
    tensor([[ 1.0000,  0.0000, -1.0000],
            [ 1.0000,  0.5000, -0.7500],
            [ 1.0000,  1.0000,  0.0000]])
    """
    # Build columns without in-place ops for autograd compatibility
    columns = []

    # He_0 = 1
    He_prev_prev = torch.ones_like(x)
    columns.append(He_prev_prev)

    if degree >= 1:
        # He_1 = x
        He_prev = x.clone()
        columns.append(He_prev)

        # Recurrence: He_{n+1} = x * He_n - n * He_{n-1}
        for n in range(1, degree):
            He_curr = x * He_prev - n * He_prev_prev
            columns.append(He_curr)
            He_prev_prev = He_prev
            He_prev = He_curr

    return torch.stack(columns, dim=-1)
