"""FEM linear system solvers."""

from __future__ import annotations

import torch
from torch import Tensor


def solve_direct(
    matrix: Tensor,
    vector: Tensor,
) -> Tensor:
    """Solve a linear system Ku = f using direct methods.

    Parameters
    ----------
    matrix : Tensor
        System matrix (sparse CSR, sparse COO, or dense), shape (n, n).
    vector : Tensor
        Right-hand side vector, shape (n,) or (n, m) for multiple RHS.

    Returns
    -------
    Tensor
        Solution vector(s), same shape as vector.

    Notes
    -----
    For sparse matrices, converts to dense before solving. This is efficient
    for small to moderate-sized problems. For large problems, use solve_cg.

    The solver is fully differentiable with respect to both matrix and vector
    inputs.

    Examples
    --------
    >>> import torch
    >>> from torchscience.finite_element_method import solve_direct
    >>> K = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    >>> f = torch.tensor([5.0, 4.0], dtype=torch.float64)
    >>> u = solve_direct(K, f)
    >>> torch.allclose(K @ u, f)
    True

    """
    # Convert sparse matrices to dense
    if matrix.is_sparse or matrix.is_sparse_csr:
        matrix_dense = matrix.to_dense()
    else:
        matrix_dense = matrix

    # Use torch.linalg.solve which has autograd support
    # solve expects (*, n, n) @ (*, n, k) -> (*, n, k)
    # or (*, n, n) @ (*, n) -> (*, n)
    return torch.linalg.solve(matrix_dense, vector)
