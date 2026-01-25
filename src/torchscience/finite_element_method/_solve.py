"""FEM linear system solvers."""

from __future__ import annotations

from typing import Callable

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


def solve_cg(
    matrix: Tensor,
    vector: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    preconditioner: Tensor | Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Solve a linear system Ku = f using conjugate gradient.

    The conjugate gradient (CG) method is an iterative algorithm for solving
    symmetric positive definite (SPD) linear systems. It is particularly
    efficient for large sparse systems where direct methods would be too
    expensive.

    Parameters
    ----------
    matrix : Tensor
        Symmetric positive definite system matrix (sparse or dense), shape (n, n).
    vector : Tensor
        Right-hand side vector, shape (n,).
    x0 : Tensor, optional
        Initial guess for the solution. Default is zeros.
    tol : float, optional
        Convergence tolerance. The algorithm stops when
        ||r|| < tol * ||b||. Default 1e-6.
    maxiter : int, optional
        Maximum number of iterations. Default is 2*n.
    preconditioner : Tensor or callable, optional
        Preconditioner to accelerate convergence. If a Tensor, it is treated
        as the diagonal of M^{-1} and applied element-wise. If callable,
        it should compute M^{-1} @ r for a given residual r.

    Returns
    -------
    Tensor
        Solution vector, shape (n,).

    Notes
    -----
    CG requires the matrix to be symmetric positive definite. For non-SPD
    systems, use GMRES or direct methods.

    The algorithm implements the standard preconditioned CG method:

    .. math::

        \\text{Initialize: } r_0 = b - A x_0, \\; z_0 = M^{-1} r_0, \\; p_0 = z_0

        \\text{For } k = 0, 1, 2, \\ldots

            \\alpha_k = \\frac{r_k^T z_k}{p_k^T A p_k}

            x_{k+1} = x_k + \\alpha_k p_k

            r_{k+1} = r_k - \\alpha_k A p_k

            \\text{if } \\|r_{k+1}\\| < \\text{tol} \\cdot \\|b\\|: \\text{ converged}

            z_{k+1} = M^{-1} r_{k+1}

            \\beta_k = \\frac{r_{k+1}^T z_{k+1}}{r_k^T z_k}

            p_{k+1} = z_{k+1} + \\beta_k p_k

    Examples
    --------
    >>> import torch
    >>> from torchscience.finite_element_method import solve_cg
    >>> K = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    >>> f = torch.tensor([5.0, 4.0], dtype=torch.float64)
    >>> u = solve_cg(K, f)
    >>> torch.allclose(K @ u, f, atol=1e-6)
    True

    """
    n = vector.shape[0]

    # Default maxiter
    if maxiter is None:
        maxiter = 2 * n

    # Initial guess
    if x0 is None:
        x = torch.zeros_like(vector)
    else:
        x = x0.clone()

    # Setup preconditioner function
    if preconditioner is None:
        # No preconditioning: M^{-1} = I
        def apply_precond(r: Tensor) -> Tensor:
            return r
    elif callable(preconditioner):
        apply_precond = preconditioner
    else:
        # Tensor: treat as diagonal of M^{-1}
        diag_inv = preconditioner

        def apply_precond(r: Tensor) -> Tensor:
            return diag_inv * r

    # Initial residual: r = b - A @ x
    r = vector - matrix @ x

    # Convergence threshold
    b_norm = vector.norm()
    threshold = tol * b_norm

    # Check if already converged
    if r.norm() < threshold:
        return x

    # Apply preconditioner: z = M^{-1} @ r
    z = apply_precond(r)

    # Initial search direction
    p = z.clone()

    # r^T @ z for the current iteration
    rz = torch.dot(r, z)

    for _ in range(maxiter):
        # Matrix-vector product: A @ p
        Ap = matrix @ p

        # Step size: alpha = (r^T @ z) / (p^T @ A @ p)
        pAp = torch.dot(p, Ap)
        alpha = rz / pAp

        # Update solution: x = x + alpha * p
        x = x + alpha * p

        # Update residual: r = r - alpha * A @ p
        r = r - alpha * Ap

        # Check convergence
        r_norm = r.norm()
        if r_norm < threshold:
            break

        # Apply preconditioner: z = M^{-1} @ r
        z = apply_precond(r)

        # New r^T @ z
        rz_new = torch.dot(r, z)

        # Direction update factor: beta = (r_{k+1}^T @ z_{k+1}) / (r_k^T @ z_k)
        beta = rz_new / rz

        # Update search direction: p = z + beta * p
        p = z + beta * p

        # Store for next iteration
        rz = rz_new

    return x
