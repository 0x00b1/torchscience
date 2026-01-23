"""Newton's method with Jacobian caching for implicit ODE solvers."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass
class JacobianCache:
    """Cache for Jacobian factorization to enable reuse across Newton iterations.

    This cache stores the LU factorization of the Jacobian matrix, allowing
    efficient reuse when the Jacobian doesn't change significantly between
    iterations (modified Newton method) or between solver steps (implicit ODE
    solvers with slowly-varying Jacobians).

    Attributes
    ----------
    lu_factors : Tensor, optional
        LU factorization of the Jacobian (L and U packed together).
    lu_pivots : Tensor, optional
        Pivot indices from LU factorization.
    jacobian : Tensor, optional
        The original Jacobian matrix (before factorization).
    n_factorizations : int
        Number of LU factorizations performed (for diagnostics).
    """

    lu_factors: Optional[torch.Tensor] = None
    lu_pivots: Optional[torch.Tensor] = None
    jacobian: Optional[torch.Tensor] = None
    n_factorizations: int = 0

    def clear(self):
        """Clear cached factorization."""
        self.lu_factors = None
        self.lu_pivots = None
        self.jacobian = None


def newton_solve_cached(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 10,
    jacobian: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    cache: Optional[JacobianCache] = None,
    recompute_jacobian_every: int = 1,
) -> Tuple[torch.Tensor, bool, Dict]:
    """
    Solve f(x) = 0 using Newton's method with Jacobian caching.

    This solver is designed for use in implicit ODE solvers (BDF, Radau) where
    the same or similar nonlinear system is solved repeatedly. It supports:

    - LU factorization caching to avoid redundant factorizations
    - Modified Newton method (reusing Jacobian across iterations)
    - User-provided Jacobian functions

    Parameters
    ----------
    f : callable
        Function to find root of. f(x) -> residual with same shape as x.
    x0 : Tensor
        Initial guess.
    tol : float
        Convergence tolerance on residual norm.
    max_iter : int
        Maximum number of Newton iterations.
    jacobian : callable, optional
        User-provided Jacobian function. If None, computed via torch.func.jacrev.
    cache : JacobianCache, optional
        Cache object for storing LU factorization. If provided, factorization
        may be reused across calls (useful for implicit ODE solvers).
    recompute_jacobian_every : int
        Recompute Jacobian every N iterations (modified Newton method).
        Set to 1 for full Newton. Larger values reduce cost but may slow convergence.

    Returns
    -------
    x : Tensor
        Solution (or last iterate if not converged).
    converged : bool
        Whether the method converged within tolerance.
    info : dict
        Diagnostic information: n_iterations, final_residual_norm.

    Examples
    --------
    >>> def f(x):
    ...     return x**2 - 2
    >>> x0 = torch.tensor([1.5], dtype=torch.float64)
    >>> x, converged, info = newton_solve_cached(f, x0)
    >>> converged
    True
    >>> torch.allclose(x, torch.sqrt(torch.tensor([2.0], dtype=torch.float64)))
    True

    Using cache for modified Newton (reuse Jacobian):
    >>> cache = JacobianCache()
    >>> x, converged, info = newton_solve_cached(
    ...     f, x0, cache=cache, recompute_jacobian_every=5
    ... )
    >>> cache.n_factorizations < info["n_iterations"]
    True
    """
    if cache is None:
        cache = JacobianCache()

    x = x0.clone()

    # Determine Jacobian computation method
    if jacobian is not None:
        compute_jacobian = jacobian
    else:
        compute_jacobian = lambda x_: torch.func.jacrev(f)(x_)

    for iteration in range(max_iter):
        residual = f(x)
        residual_flat = residual.view(-1)
        residual_norm = torch.linalg.norm(residual_flat)

        if residual_norm < tol:
            return (
                x,
                True,
                {
                    "n_iterations": iteration + 1,
                    "final_residual_norm": residual_norm.item(),
                },
            )

        # Recompute Jacobian if needed
        need_new_jacobian = (
            cache.lu_factors is None
            or iteration % recompute_jacobian_every == 0
        )

        if need_new_jacobian:
            J = compute_jacobian(x)
            # Ensure J is 2D
            if J.dim() == 1:
                J = J.unsqueeze(0)
            cache.jacobian = J

            # LU factorization
            try:
                cache.lu_factors, cache.lu_pivots = torch.linalg.lu_factor(J)
                cache.n_factorizations += 1
            except RuntimeError:
                # Singular Jacobian
                return (
                    x,
                    False,
                    {
                        "n_iterations": iteration + 1,
                        "final_residual_norm": residual_norm.item(),
                    },
                )

        # Solve J @ dx = -residual using cached LU
        try:
            dx = torch.linalg.lu_solve(
                cache.lu_factors, cache.lu_pivots, -residual_flat.unsqueeze(-1)
            ).squeeze(-1)
        except RuntimeError:
            return (
                x,
                False,
                {
                    "n_iterations": iteration + 1,
                    "final_residual_norm": residual_norm.item(),
                },
            )

        x = x + dx.view(x.shape)

    # Final convergence check
    residual = f(x)
    residual_norm = torch.linalg.norm(residual.view(-1))
    converged = residual_norm < tol

    return (
        x,
        converged,
        {
            "n_iterations": max_iter,
            "final_residual_norm": residual_norm.item(),
        },
    )
