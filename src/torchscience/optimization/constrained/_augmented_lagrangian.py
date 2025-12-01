from typing import Callable, Optional

import torch
from torch import Tensor


class _ALImplicitGrad(torch.autograd.Function):
    """
    Implicit differentiation through augmented Lagrangian optimum via KKT conditions.

    At the optimum (x*, λ*, μ*), the KKT conditions hold:
        ∇f(x*) + Jₕᵀλ* + Jᵧᵀμ* = 0  (stationarity)
        h(x*) = 0                      (primal feasibility - equality)
        g(x*) <= 0, μ* >= 0            (primal feasibility - inequality)
        μ* ⊙ g(x*) = 0                 (complementary slackness)

    Differentiating these conditions gives the implicit gradient.
    """

    @staticmethod
    def forward(
        ctx,
        result: Tensor,
        objective: Callable[[Tensor], Tensor],
        eq_constraints: Optional[Callable[[Tensor], Tensor]],
        ineq_constraints: Optional[Callable[[Tensor], Tensor]],
        lambda_eq: Optional[Tensor],
        mu_ineq: Optional[Tensor],
    ) -> Tensor:
        ctx.objective = objective
        ctx.eq_constraints = eq_constraints
        ctx.ineq_constraints = ineq_constraints
        ctx.save_for_backward(result)
        ctx.lambda_eq = lambda_eq
        ctx.mu_ineq = mu_ineq
        return result.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (result,) = ctx.saved_tensors
        lambda_eq = ctx.lambda_eq
        mu_ineq = ctx.mu_ineq

        with torch.enable_grad():
            x = result.detach().requires_grad_(True)

            # Compute gradient of Lagrangian w.r.t. x
            f = ctx.objective(x)
            grad_f = torch.autograd.grad(f, x, create_graph=True)[0]

            total_grad = grad_f.clone()

            if ctx.eq_constraints is not None and lambda_eq is not None:
                h = ctx.eq_constraints(x)
                if h.dim() == 0:
                    h = h.unsqueeze(0)
                for i in range(h.numel()):
                    grad_hi = torch.autograd.grad(
                        h.flatten()[i], x, retain_graph=True, create_graph=True
                    )[0]
                    total_grad = total_grad + lambda_eq.flatten()[i] * grad_hi

            if ctx.ineq_constraints is not None and mu_ineq is not None:
                g = ctx.ineq_constraints(x)
                if g.dim() == 0:
                    g = g.unsqueeze(0)
                for i in range(g.numel()):
                    if mu_ineq.flatten()[i] > 1e-8:  # Active constraint
                        grad_gi = torch.autograd.grad(
                            g.flatten()[i],
                            x,
                            retain_graph=True,
                            create_graph=True,
                        )[0]
                        total_grad = (
                            total_grad + mu_ineq.flatten()[i] * grad_gi
                        )

            # Compute Hessian of Lagrangian (for implicit function theorem)
            hess_rows = []
            for i in range(x.numel()):
                grad_i = torch.autograd.grad(
                    total_grad.flatten()[i],
                    x,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad_i is None:
                    grad_i = torch.zeros_like(x)
                hess_rows.append(grad_i.flatten())

            H = torch.stack(hess_rows)

            # Add regularization for numerical stability
            reg = 1e-4 * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
            H = H + reg

            # Solve H @ v = grad_output for implicit gradient
            try:
                v = torch.linalg.solve(H, grad_output.flatten())
            except RuntimeError:
                v = torch.linalg.lstsq(H, grad_output.flatten()).solution

            # Backpropagate through objective
            f.backward(-v @ grad_f)

        return None, None, None, None, None, None


def augmented_lagrangian(
    objective: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    eq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    ineq_constraints: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 50,
    inner_maxiter: int = 100,
    rho: float = 1.0,
    rho_max: float = 1e6,
) -> Tensor:
    r"""
    Augmented Lagrangian method for constrained optimization.

    Solves the constrained optimization problem:

    .. math::

        \min_x f(x) \quad \text{s.t.} \quad h(x) = 0, \; g(x) \leq 0

    by iteratively solving unconstrained subproblems with the augmented
    Lagrangian:

    .. math::

        L_\rho(x, \lambda, \mu) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\|h(x)\|^2
        + \sum_i \frac{1}{2\rho}[\max(0, \mu_i + \rho g_i(x))^2 - \mu_i^2]

    Parameters
    ----------
    objective : Callable[[Tensor], Tensor]
        Objective function f(x) to minimize. Takes parameters of shape ``(n,)``
        and returns a scalar.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    eq_constraints : Callable, optional
        Equality constraint function h(x). Returns tensor where h(x) = 0 is required.
    ineq_constraints : Callable, optional
        Inequality constraint function g(x). Returns tensor where g(x) <= 0 is required.
    tol : float, optional
        Convergence tolerance on constraint violation. Default: ``sqrt(eps)`` for dtype.
    maxiter : int
        Maximum outer iterations (Lagrange multiplier updates). Default: 50.
    inner_maxiter : int
        Maximum inner iterations per subproblem. Default: 100.
    rho : float
        Initial penalty parameter. Default: 1.0.
    rho_max : float
        Maximum penalty parameter. Default: 1e6.

    Returns
    -------
    Tensor
        Optimized parameters of shape ``(n,)``.

    Examples
    --------
    Minimize x² + y² subject to x + y = 1:

    >>> def objective(x):
    ...     return torch.sum(x**2)
    >>> def eq_constraint(x):
    ...     return x.sum() - 1.0
    >>> x0 = torch.zeros(2)
    >>> result = augmented_lagrangian(objective, x0, eq_constraints=eq_constraint)
    >>> result
    tensor([0.5, 0.5])

    With inequality constraints (x >= 0.6):

    >>> def ineq_constraint(x):
    ...     return 0.6 - x[0]  # -x + 0.6 <= 0
    >>> result = augmented_lagrangian(
    ...     objective, x0, eq_constraints=eq_constraint, ineq_constraints=ineq_constraint
    ... )
    >>> result
    tensor([0.6, 0.4])

    References
    ----------
    - Nocedal, J. and Wright, S.J. "Numerical Optimization." Chapter 17.
    - Bertsekas, D.P. "Constrained Optimization and Lagrange Multiplier Methods."

    See Also
    --------
    https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone()

    # Initialize Lagrange multipliers
    lambda_eq = None
    mu_ineq = None

    if eq_constraints is not None:
        h0 = eq_constraints(x0)
        if h0.dim() == 0:
            h0 = h0.unsqueeze(0)
        lambda_eq = torch.zeros(h0.numel(), dtype=x0.dtype, device=x0.device)

    if ineq_constraints is not None:
        g0 = ineq_constraints(x0)
        if g0.dim() == 0:
            g0 = g0.unsqueeze(0)
        mu_ineq = torch.zeros(g0.numel(), dtype=x0.dtype, device=x0.device)

    for outer_iter in range(maxiter):
        # Capture current multiplier values for closure
        current_lambda_eq = (
            lambda_eq.clone() if lambda_eq is not None else None
        )
        current_mu_ineq = mu_ineq.clone() if mu_ineq is not None else None
        current_rho = rho

        # Define augmented Lagrangian for inner minimization
        def augmented_lagrangian_func(x_inner):
            L = objective(x_inner)

            if eq_constraints is not None:
                h = eq_constraints(x_inner)
                if h.dim() == 0:
                    h = h.unsqueeze(0)
                L = (
                    L
                    + torch.dot(current_lambda_eq, h)
                    + (current_rho / 2) * torch.sum(h**2)
                )

            if ineq_constraints is not None:
                g = ineq_constraints(x_inner)
                if g.dim() == 0:
                    g = g.unsqueeze(0)
                # Powell-Hestenes-Rockafellar formulation
                for i in range(g.numel()):
                    slack = torch.clamp(
                        current_mu_ineq[i] + current_rho * g[i], min=0.0
                    )
                    L = L + (1 / (2 * current_rho)) * (
                        slack**2 - current_mu_ineq[i] ** 2
                    )

            return L

        # Inner loop: minimize augmented Lagrangian using gradient descent
        x_inner = x.clone().requires_grad_(True)

        for inner_iter in range(inner_maxiter):
            L = augmented_lagrangian_func(x_inner)
            grad = torch.autograd.grad(L, x_inner, create_graph=False)[0]

            if torch.norm(grad) < tol:
                break

            # Simple gradient descent with line search
            alpha = 1.0
            x_new = x_inner.detach() - alpha * grad

            # Backtracking line search
            for _ in range(20):
                with torch.no_grad():
                    L_new = augmented_lagrangian_func(x_new)
                    L_old = augmented_lagrangian_func(x_inner.detach())
                if L_new < L_old:
                    break
                alpha *= 0.5
                x_new = x_inner.detach() - alpha * grad

            x_inner = x_new.requires_grad_(True)

        x = x_inner.detach()

        # Check constraint satisfaction
        max_violation = 0.0

        if eq_constraints is not None:
            h = eq_constraints(x)
            if h.dim() == 0:
                h = h.unsqueeze(0)
            max_violation = max(max_violation, torch.max(torch.abs(h)).item())
            # Update equality multipliers
            lambda_eq = lambda_eq + rho * h.detach()

        if ineq_constraints is not None:
            g = ineq_constraints(x)
            if g.dim() == 0:
                g = g.unsqueeze(0)
            max_violation = max(
                max_violation, torch.max(torch.clamp(g, min=0)).item()
            )
            # Update inequality multipliers
            mu_ineq = torch.clamp(mu_ineq + rho * g.detach(), min=0.0)

        if max_violation < tol:
            break

        # Increase penalty if constraints not satisfied
        if max_violation > 0.25 * tol:
            rho = min(rho * 2, rho_max)

    # Attach implicit gradient for backpropagation
    return _ALImplicitGrad.apply(
        x, objective, eq_constraints, ineq_constraints, lambda_eq, mu_ineq
    )
