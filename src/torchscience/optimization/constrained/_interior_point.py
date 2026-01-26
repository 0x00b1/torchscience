from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult


class _IPImplicitGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_opt, objective, eq_constraints, ineq_constraints):
        ctx.objective = objective
        ctx.eq_constraints = eq_constraints
        ctx.ineq_constraints = ineq_constraints
        ctx.save_for_backward(x_opt)
        return x_opt.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (x_opt,) = ctx.saved_tensors

        with torch.enable_grad():
            x = x_opt.detach().requires_grad_(True)
            f_val = ctx.objective(x)
            if f_val.dim() > 0:
                f_val = f_val.sum()
            grad_f = torch.autograd.grad(f_val, x, create_graph=True)[0]

            # Build Hessian and solve for implicit gradient
            n = x.numel()
            hess_rows = []
            for i in range(n):
                h_row = torch.autograd.grad(
                    grad_f[i], x, retain_graph=True, create_graph=False
                )[0]
                if h_row is None:
                    h_row = torch.zeros_like(x)
                hess_rows.append(h_row.detach())
            H = torch.stack(hess_rows)
            reg = 1e-6 * torch.eye(n, dtype=H.dtype, device=H.device)
            v = torch.linalg.solve(H + reg, grad_output.detach())

        return v, None, None, None


def _barrier_merit(
    x: Tensor,
    objective: Callable[[Tensor], Tensor],
    eq_constraints: Optional[Callable[[Tensor], Tensor]],
    ineq_constraints: Optional[Callable[[Tensor], Tensor]],
    mu: float,
    rho: float,
    lambda_eq: Optional[Tensor],
) -> Tensor:
    """Compute the barrier + augmented-Lagrangian merit function.

    merit = f(x) - mu * sum(log(-g_i(x)))
            + lambda_eq^T h(x) + (rho/2) * ||h(x)||^2
    """
    val = objective(x)
    if val.dim() > 0:
        val = val.sum()

    if ineq_constraints is not None:
        g = ineq_constraints(x)
        if g.dim() == 0:
            g = g.unsqueeze(0)
        # Log-barrier: only valid when g < 0
        neg_g = -g
        # Clamp for numerical safety (should be positive in feasible region)
        val = val - mu * torch.log(neg_g.clamp(min=1e-20)).sum()

    if eq_constraints is not None:
        h = eq_constraints(x)
        if h.dim() == 0:
            h = h.unsqueeze(0)
        # Augmented Lagrangian terms for equality constraints
        if lambda_eq is not None:
            val = val + torch.dot(lambda_eq, h)
        val = val + (rho / 2.0) * torch.sum(h**2)

    return val


def interior_point(
    objective: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    eq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    ineq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 100,
    mu_init: float = 0.1,
    mu_factor: float = 0.2,
) -> OptimizeResult:
    r"""
    Primal-dual interior point method for constrained optimization.

    Solves: min f(x) s.t. h(x) = 0, g(x) <= 0

    Uses a log-barrier for inequality constraints and an augmented
    Lagrangian for equality constraints, with Newton steps on the
    barrier subproblem. The barrier parameter mu decreases toward zero.

    Parameters
    ----------
    objective : Callable[[Tensor], Tensor]
        Scalar-valued objective function.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    eq_constraints : Callable[[Tensor], Tensor], optional
        Equality constraints h(x) = 0. Returns tensor of shape ``(m_eq,)``.
    ineq_constraints : Callable[[Tensor], Tensor], optional
        Inequality constraints g(x) <= 0. Returns tensor of shape ``(m_ineq,)``.
    tol : float, optional
        KKT tolerance. Default: ``torch.finfo(x0.dtype).eps ** 0.5``.
    maxiter : int, optional
        Maximum outer iterations. Default: 100.
    mu_init : float, optional
        Initial barrier parameter. Default: 0.1.
    mu_factor : float, optional
        Barrier reduction factor per outer iteration. Default: 0.2.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``, ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone().detach().to(dtype=x0.dtype)
    n = x.numel()
    mu = mu_init
    num_iter = 0

    # Initialize equality dual variables and penalty
    lambda_eq = None
    rho = 1.0
    if eq_constraints is not None:
        with torch.no_grad():
            h0 = eq_constraints(x)
        if h0.dim() == 0:
            h0 = h0.unsqueeze(0)
        lambda_eq = torch.zeros(h0.numel(), dtype=x0.dtype, device=x0.device)

    converged = torch.tensor(False, device=x0.device)

    for outer in range(maxiter):
        num_iter = outer + 1

        # Inner loop: minimize the barrier + augmented Lagrangian merit
        # function for the current mu and lambda_eq
        current_mu = mu
        current_lambda_eq = (
            lambda_eq.clone() if lambda_eq is not None else None
        )
        current_rho = rho

        for inner in range(50):
            x_g = x.detach().requires_grad_(True)
            merit = _barrier_merit(
                x_g,
                objective,
                eq_constraints,
                ineq_constraints,
                current_mu,
                current_rho,
                current_lambda_eq,
            )
            grad_merit = torch.autograd.grad(merit, x_g)[0]
            grad_merit = grad_merit.detach()

            if grad_merit.abs().max().item() < tol * 0.1:
                break

            # Compute Hessian of merit function for Newton step
            x_h = x.detach().requires_grad_(True)
            merit_h = _barrier_merit(
                x_h,
                objective,
                eq_constraints,
                ineq_constraints,
                current_mu,
                current_rho,
                current_lambda_eq,
            )
            grad_h = torch.autograd.grad(merit_h, x_h, create_graph=True)[0]
            hess_rows = []
            for i in range(n):
                row = torch.autograd.grad(grad_h[i], x_h, retain_graph=True)[0]
                if row is None:
                    row = torch.zeros_like(x)
                hess_rows.append(row.detach())
            H = torch.stack(hess_rows)

            # Regularize Hessian to be positive definite
            reg = 1e-4 * torch.eye(n, dtype=H.dtype, device=H.device)
            H = H + reg

            # Newton direction: H @ d = -grad
            try:
                direction = torch.linalg.solve(H, -grad_merit)
            except RuntimeError:
                direction = -grad_merit  # Fall back to gradient descent

            # Backtracking line search on the merit function
            alpha = 1.0
            merit_current = merit.detach()
            armijo_slope = torch.dot(grad_merit, direction).item()

            if armijo_slope > 0:
                # Not a descent direction, use negative gradient
                direction = -grad_merit
                armijo_slope = torch.dot(grad_merit, direction).item()

            for _ in range(30):
                x_trial = x + alpha * direction

                # Check feasibility for inequality constraints
                feasible = True
                if ineq_constraints is not None:
                    with torch.no_grad():
                        g_trial = ineq_constraints(x_trial)
                        if g_trial.dim() == 0:
                            g_trial = g_trial.unsqueeze(0)
                        if (g_trial >= 0).any():
                            feasible = False

                if feasible:
                    with torch.no_grad():
                        merit_trial = _barrier_merit(
                            x_trial,
                            objective,
                            eq_constraints,
                            ineq_constraints,
                            current_mu,
                            current_rho,
                            current_lambda_eq,
                        )
                    # Armijo condition: sufficient decrease
                    if (
                        merit_trial
                        <= merit_current + 1e-4 * alpha * armijo_slope
                    ):
                        break

                alpha *= 0.5
                if alpha < 1e-16:
                    break

            x = (x + alpha * direction).detach()

        # Update equality multipliers (augmented Lagrangian update)
        h_val = None
        if eq_constraints is not None:
            with torch.no_grad():
                h_val = eq_constraints(x)
                if h_val.dim() == 0:
                    h_val = h_val.unsqueeze(0)
                lambda_eq = lambda_eq + rho * h_val
                # Increase penalty if constraints still violated
                if h_val.abs().max().item() > tol:
                    rho = min(rho * 2.0, 1e6)

        # Check convergence
        # Compute objective gradient for KKT check
        x_check = x.detach().requires_grad_(True)
        f_check = objective(x_check)
        if f_check.dim() > 0:
            f_check = f_check.sum()
        grad_f_check = torch.autograd.grad(f_check, x_check)[0].detach()

        kkt_residual = grad_f_check.abs().max().item()

        eq_violation = 0.0
        if eq_constraints is not None and h_val is not None:
            eq_violation = h_val.abs().max().item()

        ineq_violation = 0.0
        g_val = None
        if ineq_constraints is not None:
            with torch.no_grad():
                g_val = ineq_constraints(x)
                if g_val.dim() == 0:
                    g_val = g_val.unsqueeze(0)
                ineq_violation = g_val.clamp(min=0).max().item()

        total_violation = max(eq_violation, ineq_violation)

        if total_violation < tol and mu < tol:
            converged = torch.tensor(True, device=x0.device)
            break

        # Reduce barrier parameter
        mu = mu * mu_factor

    else:
        # Check if we converged even on the last iteration
        if total_violation < tol:
            converged = torch.tensor(True, device=x0.device)
        else:
            converged = torch.tensor(False, device=x0.device)

    with torch.no_grad():
        f_final = objective(x)

    x_with_grad = _IPImplicitGrad.apply(
        x, objective, eq_constraints, ineq_constraints
    )

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=x0.device
        ),
        fun=f_final,
    )
