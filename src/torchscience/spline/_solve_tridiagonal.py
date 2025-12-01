import torch
from torch import Tensor


def solve_tridiagonal(
    diag: Tensor,
    upper: Tensor,
    lower: Tensor,
    rhs: Tensor,
) -> Tensor:
    """
    Solve a tridiagonal system Ax = b using the Thomas algorithm.

    The matrix A has the form:
        [d0  u0   0   0  ...  0   0 ]
        [l0  d1  u1   0  ...  0   0 ]
        [ 0  l1  d2  u2  ...  0   0 ]
        [        ...                ]
        [ 0   0   0   0  ... ln-2 dn-1]

    Parameters
    ----------
    diag : Tensor
        Main diagonal, shape (n,)
    upper : Tensor
        Upper diagonal, shape (n-1,)
    lower : Tensor
        Lower diagonal, shape (n-1,)
    rhs : Tensor
        Right-hand side, shape (*batch, n)

    Returns
    -------
    Tensor
        Solution x, shape (*batch, n)

    Notes
    -----
    This implementation is fully differentiable. The backward pass
    solves another tridiagonal system (the adjoint) with O(n) complexity.
    """
    n = diag.shape[0]

    # Move system axis to front for easier indexing
    # rhs: (*batch, n) -> (n, *batch)
    rhs_t = rhs.movedim(-1, 0)

    # Forward elimination using lists to avoid in-place operations
    c_prime_list = []
    d_prime_list = []

    c_prime_list.append(upper[0] / diag[0])
    d_prime_list.append(rhs_t[0] / diag[0])

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime_list[i - 1]
        c_prime_list.append(upper[i] / denom)
        d_prime_list.append(
            (rhs_t[i] - lower[i - 1] * d_prime_list[i - 1]) / denom
        )

    # Last row (no upper diagonal)
    denom = diag[n - 1] - lower[n - 2] * c_prime_list[n - 2]
    d_prime_list.append(
        (rhs_t[n - 1] - lower[n - 2] * d_prime_list[n - 2]) / denom
    )

    # Back substitution using list to avoid in-place operations
    # Build solution from back to front
    x_list = [None] * n
    x_list[n - 1] = d_prime_list[n - 1]

    for i in range(n - 2, -1, -1):
        x_list[i] = d_prime_list[i] - c_prime_list[i] * x_list[i + 1]

    # Stack results along the first dimension
    x = torch.stack(x_list, dim=0)

    # Move system axis back: (n, *batch) -> (*batch, n)
    return x.movedim(0, -1)
