"""Newton's method for solving nonlinear systems."""

from typing import Callable, Tuple

import torch


def newton_solve(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 10,
) -> Tuple[torch.Tensor, bool]:
    """
    Solve f(x) = 0 using Newton's method with automatic Jacobian.

    Parameters
    ----------
    f : callable
        Function to find root of. f(x) -> residual with same shape as x.
    x0 : Tensor
        Initial guess. Shape (*batch, n) for n-dimensional system.
    tol : float
        Convergence tolerance on residual norm.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    x : Tensor
        Solution (or last iterate if not converged).
    converged : bool
        Whether the method converged within tolerance.
    """
    x = x0.clone()
    original_shape = x.shape

    # Handle 1D case: treat as single element, not as batch dimension
    if x.dim() == 1:
        # Shape (n,) -> single system with n variables
        n = x.shape[0]

        for _ in range(max_iter):
            residual = f(x)
            residual_norm = torch.linalg.norm(residual)

            if residual_norm < tol:
                return x, True

            # Compute Jacobian
            jac = torch.func.jacrev(f)(x)  # (n, n) for n-dim system

            # Handle scalar case: Jacobian is (1,) not (1, 1)
            if jac.dim() == 1:
                jac = jac.unsqueeze(0)

            try:
                dx = torch.linalg.solve(jac, -residual.unsqueeze(-1)).squeeze(
                    -1
                )
            except RuntimeError:
                return x, False

            x = x + dx

        # Final convergence check
        residual = f(x)
        converged = torch.linalg.norm(residual) < tol
        return x, bool(converged)

    # Handle 2D batched case: shape (batch, n)
    batch_size = x.shape[0]
    n = x.shape[1]

    for _ in range(max_iter):
        residual = f(x)  # (batch, n)
        residual_norm = torch.linalg.norm(residual, dim=-1)  # (batch,)

        if (residual_norm < tol).all():
            return x, True

        # Compute Jacobian for each batch element
        jacobians = []
        for i in range(batch_size):

            def f_i(xi):
                # Create full batch input with only element i varying
                x_copy = x.clone()
                x_copy[i] = xi
                result = f(x_copy)
                return result[i]

            jac = torch.func.jacrev(f_i)(x[i])  # (n, n)
            if jac.dim() == 1:
                jac = jac.unsqueeze(0)
            jacobians.append(jac)

        J = torch.stack(jacobians)  # (batch, n, n)

        try:
            dx = torch.linalg.solve(J, -residual.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            return x, False

        x = x + dx

    # Final convergence check
    residual = f(x)
    residual_norm = torch.linalg.norm(residual, dim=-1)
    converged = (residual_norm < tol).all()
    return x, bool(converged)
