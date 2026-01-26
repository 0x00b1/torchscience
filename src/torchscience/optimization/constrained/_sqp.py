from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import _compute_grad


def _solve_qp_equality(H, c, A, b):
    """Solve min 0.5 x^T H x + c^T x  s.t. A x = b via KKT system."""
    n = H.shape[0]
    m = A.shape[0] if A is not None and A.numel() > 0 else 0

    if m == 0:
        # Unconstrained QP
        reg = 1e-8 * torch.eye(n, dtype=H.dtype, device=H.device)
        return torch.linalg.solve(H + reg, -c)

    # KKT system: [[H, A^T], [A, 0]] [x; lambda] = [-c; b]
    KKT = torch.zeros(n + m, n + m, dtype=H.dtype, device=H.device)
    KKT[:n, :n] = H
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A

    rhs = torch.zeros(n + m, dtype=H.dtype, device=H.device)
    rhs[:n] = -c
    rhs[n:] = b

    reg = 1e-8 * torch.eye(n + m, dtype=H.dtype, device=H.device)
    sol = torch.linalg.solve(KKT + reg, rhs)
    return sol[:n]


def sqp(
    objective: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    eq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    ineq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 50,
) -> OptimizeResult:
    r"""
    Sequential Quadratic Programming for constrained optimization.

    Solves: min f(x) s.t. h(x) = 0, g(x) <= 0

    At each iteration, solves a QP subproblem with a quadratic model of
    the objective and linearized constraints. Uses damped BFGS for the
    Hessian approximation and L1 merit function for step acceptance.

    Parameters
    ----------
    objective : Callable[[Tensor], Tensor]
        Scalar-valued objective function.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    eq_constraints : Callable[[Tensor], Tensor], optional
        Equality constraints h(x) = 0.
    ineq_constraints : Callable[[Tensor], Tensor], optional
        Inequality constraints g(x) <= 0.
    tol : float, optional
        KKT tolerance. Default: ``torch.finfo(x0.dtype).eps ** 0.5``.
    maxiter : int, optional
        Maximum iterations. Default: 50.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``, ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    n = x0.numel()
    x = x0.clone().detach()
    B = torch.eye(n, dtype=x0.dtype, device=x0.device)  # Hessian approx
    num_iter = 0
    penalty = 1.0

    for k in range(maxiter):
        num_iter = k + 1

        # Evaluate objective and gradient
        f_val, grad_f = _compute_grad(objective, x)

        # Evaluate constraints
        h_val = None
        J_h = None
        if eq_constraints is not None:
            h_val = eq_constraints(x.detach())
            if h_val.dim() == 0:
                h_val = h_val.unsqueeze(0)
            J_h = torch.func.jacrev(lambda xx: eq_constraints(xx).reshape(-1))(
                x.detach()
            )

        g_val = None
        J_g = None
        if ineq_constraints is not None:
            g_val = ineq_constraints(x.detach())
            if g_val.dim() == 0:
                g_val = g_val.unsqueeze(0)
            J_g = torch.func.jacrev(
                lambda xx: ineq_constraints(xx).reshape(-1)
            )(x.detach())

        # Check KKT convergence
        kkt_norm = grad_f.abs().max().item()
        if h_val is not None:
            kkt_norm = max(kkt_norm, h_val.abs().max().item())
        if g_val is not None:
            kkt_norm = max(kkt_norm, g_val.clamp(min=0).max().item())

        if kkt_norm < tol:
            converged = torch.tensor(True, device=x0.device)
            break

        # Build and solve QP subproblem
        # Convert active inequalities to equality constraints for the QP
        if ineq_constraints is not None and eq_constraints is not None:
            active = g_val >= -tol
            if active.any():
                A_active = torch.cat([J_h, J_g[active]], dim=0)
                b_active = torch.cat([-h_val, -g_val[active]], dim=0)
            else:
                A_active = J_h
                b_active = -h_val
        elif eq_constraints is not None:
            A_active = J_h
            b_active = -h_val
        elif ineq_constraints is not None:
            active = g_val >= -tol
            if active.any():
                A_active = J_g[active]
                b_active = -g_val[active]
            else:
                A_active = None
                b_active = None
        else:
            A_active = None
            b_active = None

        d = _solve_qp_equality(B, grad_f, A_active, b_active)

        # L1 merit function line search
        # Update penalty to be at least as large as the dual variables
        # This ensures the merit function is exact for the current problem
        if A_active is not None and A_active.numel() > 0:
            try:
                # Estimate dual variables from QP solution
                dual_est = torch.linalg.lstsq(
                    A_active.T, -(grad_f + B @ d)
                ).solution
                penalty = max(penalty, dual_est.abs().max().item() * 1.1)
            except Exception:
                penalty = max(penalty, 10.0)

        def merit(xx, pen):
            with torch.no_grad():
                m_val = objective(xx)
                if eq_constraints is not None:
                    h = eq_constraints(xx)
                    if h.dim() == 0:
                        h = h.unsqueeze(0)
                    m_val = m_val + pen * h.abs().sum()
                if ineq_constraints is not None:
                    g = ineq_constraints(xx)
                    if g.dim() == 0:
                        g = g.unsqueeze(0)
                    m_val = m_val + pen * g.clamp(min=0).sum()
                return m_val

        merit_current = merit(x, penalty)

        # Directional derivative of merit function
        dir_deriv = grad_f @ d
        if eq_constraints is not None:
            dir_deriv = dir_deriv - penalty * h_val.abs().sum()
        if ineq_constraints is not None:
            dir_deriv = dir_deriv - penalty * g_val.clamp(min=0).sum()

        alpha = 1.0
        for _ in range(30):
            x_new = x + alpha * d
            merit_new = merit(x_new, penalty)
            if merit_new <= merit_current + 1e-4 * alpha * dir_deriv:
                break
            alpha *= 0.5
        else:
            x_new = x + alpha * d

        # BFGS update
        _, grad_f_new = _compute_grad(objective, x_new)
        s = x_new - x
        y = grad_f_new - grad_f
        ys = (y * s).sum()

        if ys > 1e-10:
            Bs = B @ s
            sBs = (s * Bs).sum()
            B = (
                B
                - torch.outer(Bs, Bs) / sBs.clamp(min=1e-30)
                + torch.outer(y, y) / ys
            )
        else:
            # Damped BFGS: ensure positive definiteness
            Bs = B @ s
            sBs = (s * Bs).sum()
            if sBs > 1e-10:
                theta = 1.0
                if ys < 0.2 * sBs:
                    theta = (0.8 * sBs / (sBs - ys)).item()
                r = theta * y + (1 - theta) * Bs
                rs = (r * s).sum()
                if rs > 1e-10:
                    B = (
                        B
                        - torch.outer(Bs, Bs) / sBs.clamp(min=1e-30)
                        + torch.outer(r, r) / rs
                    )

        x = x_new.detach()
    else:
        converged = torch.tensor(False, device=x0.device)

    with torch.no_grad():
        f_final = objective(x)

    return OptimizeResult(
        x=x,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=x0.device
        ),
        fun=f_final,
    )
