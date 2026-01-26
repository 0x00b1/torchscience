from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import (
    _compute_grad,
    _two_loop_recursion,
)


class _LBFGSBImplicitGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_opt, fun, lower, upper):
        ctx.fun = fun
        ctx.save_for_backward(x_opt, lower, upper)
        return x_opt.clone()

    @staticmethod
    def backward(ctx, grad_output):
        x_opt, lower, upper = ctx.saved_tensors

        with torch.enable_grad():
            x = x_opt.detach().requires_grad_(True)
            f_val = ctx.fun(x)
            if f_val.dim() > 0:
                f_val = f_val.sum()
            g = torch.autograd.grad(f_val, x, create_graph=True)[0]

        # For free variables (not at bounds), implicit diff via Hessian
        # For active bounds, gradient passes through the bound parameter
        at_lower = (lower is not None) & (x_opt <= lower + 1e-8)
        at_upper = (upper is not None) & (x_opt >= upper - 1e-8)
        free = ~(at_lower | at_upper)

        # Simple approach: pass through gradient for free vars,
        # zero gradient for bound-active vars (gradient w.r.t. x is zero at bound)
        grad_x = torch.where(free, grad_output, torch.zeros_like(grad_output))

        # Gradient w.r.t. bounds
        grad_lower = (
            torch.where(at_lower, grad_output, torch.zeros_like(grad_output))
            if lower is not None
            else None
        )
        grad_upper = (
            torch.where(at_upper, grad_output, torch.zeros_like(grad_output))
            if upper is not None
            else None
        )

        return grad_x, None, grad_lower, grad_upper


def l_bfgs_b(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    lower: Optional[Tensor] = None,
    upper: Optional[Tensor] = None,
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    maxiter: int = 100,
    tol: Optional[float] = None,
    history_size: int = 10,
) -> OptimizeResult:
    r"""
    L-BFGS-B: L-BFGS with box bound constraints.

    Minimizes a scalar-valued function subject to simple bound constraints
    ``lower <= x <= upper``.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Scalar-valued objective function to minimize.
    x0 : Tensor
        Initial guess of shape ``(n,)``.
    lower : Tensor, optional
        Lower bounds of shape ``(n,)``. ``None`` means no lower bound.
    upper : Tensor, optional
        Upper bounds of shape ``(n,)``. ``None`` means no upper bound.
    grad : Callable[[Tensor], Tensor], optional
        Gradient function. If ``None``, uses ``torch.autograd``.
    maxiter : int, optional
        Maximum number of iterations. Default: 100.
    tol : float, optional
        Projected gradient norm tolerance. Default:
        ``torch.finfo(x0.dtype).eps ** 0.5``.
    history_size : int, optional
        Number of correction pairs to store. Default: 10.

    Returns
    -------
    OptimizeResult
        Result with fields ``x``, ``converged``, ``num_iterations``,
        ``fun``.
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    def _project(x):
        if lower is not None:
            x = torch.max(x, lower.detach())
        if upper is not None:
            x = torch.min(x, upper.detach())
        return x

    x = _project(x0.clone().detach())
    f_val, g = _compute_grad(fun, x, grad)

    s_history = []
    y_history = []
    rho_history = []
    num_iter = 0

    for k in range(maxiter):
        # Projected gradient for convergence check
        pg = x - _project(x - g)
        pg_inf = pg.abs().max()

        if pg_inf < tol:
            converged = torch.tensor(True, device=x.device)
            break

        num_iter = k + 1

        # Compute search direction via L-BFGS two-loop recursion
        direction = -_two_loop_recursion(g, s_history, y_history, rho_history)

        # Projected backtracking line search
        grad_dot_dir = (g * direction).sum()
        if grad_dot_dir >= 0:
            direction = -g
            grad_dot_dir = -(g * g).sum()

        alpha = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        # Projected backtracking: the _project call ensures feasibility,
        # so we do not need to compute alpha_max explicitly.
        # Use a modified Armijo condition based on the actual step taken.
        for _ in range(30):
            x_new = _project(x + alpha * direction)
            with torch.no_grad():
                f_new = fun(x_new)
            # Armijo condition using projected step
            actual_step = x_new - x
            predicted_decrease = (g * actual_step).sum()
            if f_new <= f_val + 1e-4 * predicted_decrease:
                break
            alpha = alpha * 0.5
        else:
            x_new = _project(x + alpha * direction)
            f_new, _ = _compute_grad(fun, x_new, grad)

        f_new, g_new = _compute_grad(fun, x_new, grad)

        # Update history
        s = x_new - x
        y = g_new - g
        ys = (y * s).sum()
        if ys.item() > 1e-10:
            rho_val = 1.0 / ys
            s_history.append(s)
            y_history.append(y)
            rho_history.append(rho_val)

        if len(s_history) > history_size:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)

        x = x_new
        g = g_new
        f_val = f_new
    else:
        pg = x - _project(x - g)
        converged = torch.as_tensor(pg.abs().max() < tol, device=x.device)

    with torch.no_grad():
        f_final = fun(x)

    # Attach implicit gradient
    lower_for_grad = (
        lower if lower is not None else torch.full_like(x, float("-inf"))
    )
    upper_for_grad = (
        upper if upper is not None else torch.full_like(x, float("inf"))
    )
    x_with_grad = _LBFGSBImplicitGrad.apply(
        x, fun, lower_for_grad, upper_for_grad
    )

    return OptimizeResult(
        x=x_with_grad,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=x.device
        ),
        fun=f_final,
    )
