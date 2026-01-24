"""Implicit adjoint integration for stiff systems.

Uses BDF (Backward Differentiation Formula) methods to handle
stiff adjoint ODEs that arise from stiff forward problems.
"""

import warnings
from typing import Callable, List, Tuple

import torch


def compute_jacobian(f: Callable, t: float, y: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian df/dy at (t, y).

    Computes the Jacobian matrix J[i,j] = df_i/dy_j at point (t, y).
    Uses row-by-row backward-mode differentiation.
    """
    n = y.numel()
    # Properly detach and clone to create a fresh tensor that can track grads
    y_flat = y.detach().flatten().clone().requires_grad_(True)

    with torch.enable_grad():
        f_val = f(t, y_flat.reshape(y.shape)).flatten()

    J = torch.zeros(n, n, device=y.device, dtype=y.dtype)
    for i in range(n):
        # Check if this output depends on y_flat
        if f_val[i].grad_fn is not None:
            grad_i = torch.autograd.grad(
                f_val[i],
                y_flat,
                retain_graph=(i < n - 1),
                allow_unused=True,
                create_graph=False,
            )[0]
            if grad_i is not None:
                J[i] = grad_i.detach()

    return J


def solve_implicit_system(
    A: torch.Tensor,
    b: torch.Tensor,
    solver: str = "auto",
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Solve Ax = b for implicit adjoint step.

    Parameters
    ----------
    A : Tensor
        System matrix (n, n).
    b : Tensor
        RHS vector (n,).
    solver : str
        'auto', 'direct', or 'gmres'.
    tol : float
        Tolerance for iterative solver.

    Returns
    -------
    x : Tensor
        Solution vector.
    """
    n = b.numel()

    if solver == "auto":
        solver = "direct" if n <= 1000 else "gmres"

    if solver == "direct":
        try:
            return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            warnings.warn(
                "Direct solve failed (singular matrix?). Using least-squares.",
                RuntimeWarning,
            )
            x, _, _, _ = torch.linalg.lstsq(A, b.unsqueeze(-1))
            return x.squeeze(-1)

    elif solver == "gmres":
        return _gmres(A, b, tol=tol, max_iter=min(n, 100))

    raise ValueError(f"Unknown solver: {solver}")


def _gmres(
    A: torch.Tensor, b: torch.Tensor, tol: float = 1e-6, max_iter: int = 100
) -> torch.Tensor:
    """Minimal GMRES implementation."""
    x = torch.zeros_like(b)
    r = b - A @ x
    b_norm = b.norm()

    if b_norm < 1e-14:
        return x

    if r.norm() < tol * b_norm:
        return x

    # Arnoldi process
    Q = [r / r.norm()]
    H = torch.zeros(max_iter + 1, max_iter, device=b.device, dtype=b.dtype)
    n = b.numel()

    for j in range(min(max_iter, n)):
        v = A @ Q[j]

        for i in range(j + 1):
            H[i, j] = torch.dot(Q[i], v)
            v = v - H[i, j] * Q[i]

        H[j + 1, j] = v.norm()

        if H[j + 1, j] < 1e-12:
            break

        Q.append(v / H[j + 1, j])

        # Solve least squares
        e1 = torch.zeros(j + 2, device=b.device, dtype=b.dtype)
        e1[0] = r.norm()
        y, _, _, _ = torch.linalg.lstsq(H[: j + 2, : j + 1], e1.unsqueeze(-1))

        # Build solution
        x = torch.zeros_like(b)
        for i in range(j + 1):
            x = x + y[i, 0] * Q[i]

        if (b - A @ x).norm() < tol * b_norm:
            break

    return x


def bdf1_adjoint_step(
    a: torch.Tensor,
    t: float,
    dt: float,
    f: Callable,
    y_interp: Callable,
    params: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    BDF-1 (Backward Euler) adjoint step.

    Solves: (I - dt * J^T) @ a_new = a

    Parameters
    ----------
    a : Tensor
        Current adjoint state.
    t : float
        Current time.
    dt : float
        Step size (positive).
    f : Callable
        Dynamics function.
    y_interp : Callable
        Interpolant for forward trajectory.
    params : list
        Parameters for gradient computation.

    Returns
    -------
    a_new : Tensor
        Adjoint at t - dt.
    param_grads : list
        Parameter gradient contributions.
    """
    t_next = t - dt
    y_at_t = y_interp(t_next)

    # Compute Jacobian at new time
    J = compute_jacobian(f, t_next, y_at_t)

    # System matrix: I - dt * J^T
    n = a.numel()
    I = torch.eye(n, device=a.device, dtype=a.dtype)
    A = I - dt * J.T

    # Solve for new adjoint
    a_new = solve_implicit_system(A, a.flatten())
    a_new = a_new.reshape(a.shape)

    # Compute parameter gradients
    param_grads = []
    with torch.enable_grad():
        y_local = y_at_t.clone().requires_grad_(True)
        f_val = f(t_next, y_local)

    for p in params:
        if p.requires_grad and f_val.requires_grad:
            vjp_p = torch.autograd.grad(
                f_val,
                p,
                grad_outputs=a_new.reshape(f_val.shape),
                retain_graph=True,
                allow_unused=True,
            )[0]
            if vjp_p is not None:
                param_grads.append(dt * vjp_p)
            else:
                param_grads.append(torch.zeros_like(p))
        else:
            param_grads.append(torch.zeros_like(p))

    return a_new, param_grads
