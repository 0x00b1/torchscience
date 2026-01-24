"""Newton's method for solving nonlinear systems."""

from typing import Callable, Tuple

import torch


def _reduce_norm(r: torch.Tensor) -> torch.Tensor:
    """Return 2-norm per-system (handles scalar, vector, and batched)."""
    if r.dim() == 0:
        return r.abs()
    if r.dim() == 1:
        return torch.linalg.norm(r)
    return torch.linalg.norm(r, dim=-1)


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

    # Scalar/vector (single system): shape (n,)
    if x.dim() == 1:
        for _ in range(max_iter):
            r = f(x)
            if _reduce_norm(r) < tol:
                return x, True

            J = torch.func.jacrev(f)(x)
            if J.dim() == 1:  # scalar case
                J = J.unsqueeze(0)
            try:
                dx = torch.linalg.solve(J, -r.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                return x, False

            # Simple backtracking to improve robustness on stiff problems
            step = 1.0
            old_norm = _reduce_norm(r)
            for _bt in range(5):
                x_trial = x + step * dx
                r_trial = f(x_trial)
                if _reduce_norm(r_trial) <= 0.9 * old_norm:
                    x = x_trial
                    break
                step *= 0.5
            else:
                # Fall back to full step if no improvement found
                x = x + dx

        r = f(x)
        return x, bool(_reduce_norm(r) < tol)

    # Batched systems: shape (batch, n)
    # Solve each system independently to avoid broadcasting/capture pitfalls.
    B = x.shape[0]
    solutions = []
    converged_flags = []

    for i in range(B):
        xi = x[i]

        def f_i(xi_local: torch.Tensor) -> torch.Tensor:
            X = torch.zeros_like(x)
            X[i] = xi_local
            return f(X)[i]

        sol_i, ok = newton_solve(f_i, xi, tol=tol, max_iter=max_iter)
        solutions.append(sol_i)
        converged_flags.append(ok)

    return torch.stack(solutions, dim=0), bool(all(converged_flags))
