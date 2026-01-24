"""Newton solver for BVP systems."""

from typing import Callable, Tuple

import torch
from torch import Tensor

from torchscience.ordinary_differential_equation._bvp_collocation import (
    compute_collocation_residual,
)


def newton_bvp(
    fun: Callable[[Tensor, Tensor, Tensor], Tensor],
    bc: Callable[[Tensor, Tensor, Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    p: Tensor,
    *,
    tol: float = 1e-6,
    max_iter: int = 10,
    armijo_c: float = 1e-4,
    max_linesearch: int = 10,
) -> Tuple[Tensor, Tensor, bool, int]:
    """Solve BVP system using Newton iteration with line search.

    Parameters
    ----------
    fun : callable
        RHS of ODE: dy/dx = fun(x, y, p).
    bc : callable
        Boundary conditions: bc(y(a), y(b), p) = 0.
    x : Tensor
        Mesh nodes, shape (n_nodes,).
    y : Tensor
        Initial guess for solution, shape (n_components, n_nodes).
    p : Tensor
        Initial guess for parameters, shape (n_params,).
    tol : float
        Convergence tolerance for residual norm.
    max_iter : int
        Maximum Newton iterations.
    armijo_c : float
        Armijo condition constant for line search.
    max_linesearch : int
        Maximum line search iterations.

    Returns
    -------
    y : Tensor
        Solution values, shape (n_components, n_nodes).
    p : Tensor
        Solved parameters, shape (n_params,).
    converged : bool
        Whether Newton iteration converged.
    n_iter : int
        Number of iterations taken.
    """
    n_components, n_nodes = y.shape
    n_params = p.shape[0] if p.numel() > 0 else 0
    n_intervals = n_nodes - 1

    # Total unknowns: y values at all nodes + parameters
    # But we solve for y at interior nodes + parameters
    # (boundary values are constrained by BCs)
    n_y_unknowns = n_components * n_nodes
    n_unknowns = n_y_unknowns + n_params

    def pack(y_flat: Tensor, p_flat: Tensor) -> Tensor:
        """Pack y and p into a single vector."""
        if p_flat.numel() > 0:
            return torch.cat([y_flat, p_flat])
        return y_flat

    def unpack(z: Tensor) -> Tuple[Tensor, Tensor]:
        """Unpack z into y and p."""
        y_flat = z[:n_y_unknowns].reshape(n_components, n_nodes)
        p_flat = z[n_y_unknowns:] if n_params > 0 else p
        return y_flat, p_flat

    def residual_fn(z: Tensor) -> Tensor:
        """Compute full residual vector [collocation; boundary]."""
        y_curr, p_curr = unpack(z)

        # Collocation residuals: (n_components, n_intervals)
        coll_res = compute_collocation_residual(fun, x, y_curr, p_curr)
        coll_flat = coll_res.flatten()  # (n_components * n_intervals,)

        # Boundary residuals
        ya = y_curr[:, 0]
        yb = y_curr[:, -1]
        bc_res = bc(ya, yb, p_curr)  # (n_bc,)

        return torch.cat([coll_flat, bc_res])

    # Initial state
    y_curr = y.clone()
    p_curr = p.clone()
    z = pack(y_curr.flatten(), p_curr)

    for iteration in range(max_iter):
        # Compute residual and check convergence
        res = residual_fn(z)
        res_norm = torch.linalg.norm(res)

        if res_norm < tol:
            y_sol, p_sol = unpack(z)
            return y_sol, p_sol, True, iteration + 1

        # Compute Jacobian using torch.func.jacrev
        # This is more memory-efficient than building the full Jacobian
        jac = torch.func.jacrev(residual_fn)(z)

        # Solve J * delta = -res
        try:
            # Use least squares for potentially overdetermined systems
            delta, *_ = torch.linalg.lstsq(jac, -res.unsqueeze(-1))
            delta = delta.squeeze(-1)
        except RuntimeError:
            # Jacobian is singular
            y_sol, p_sol = unpack(z)
            return y_sol, p_sol, False, iteration + 1

        # Backtracking line search (Armijo condition)
        alpha = 1.0
        z_new = z + alpha * delta
        res_new = residual_fn(z_new)
        res_new_norm = torch.linalg.norm(res_new)

        for _ in range(max_linesearch):
            if res_new_norm <= res_norm * (1 - armijo_c * alpha):
                break
            alpha *= 0.5
            z_new = z + alpha * delta
            res_new = residual_fn(z_new)
            res_new_norm = torch.linalg.norm(res_new)

        z = z_new

    # Check final convergence
    res = residual_fn(z)
    res_norm = torch.linalg.norm(res)
    y_sol, p_sol = unpack(z)
    converged = res_norm < tol

    return y_sol, p_sol, converged, max_iter
